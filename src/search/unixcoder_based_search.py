import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from search_models.unixcoder import UniXcoder

# METHODS_CSV = "/data/zxl/Search2026/outputData/repoSummaryOut/mrjob/1111/methods.csv"
# FILTERED_FILE = "/data/zxl/Search2026/outputData/repoSummaryOut/mrjob/1111/filtered.jsonl"

# METHODS_CSV = "/home/riverbag/testRepoSummaryOut/boto/boto_testAug/1122_codet5/methods.csv" 
# FILTERED_FILE = "/home/riverbag/testRepoSummaryOut/boto/boto_testAug/1122_codet5/filtered.jsonl"

METHODS_CSV = "/data/data_public/riverbag/testRepoSummaryOut/Filited/mrjob/methods.csv" 
FILTERED_FILE = "/data/data_public/riverbag/testRepoSummaryOut/Filited/mrjob/filtered.jsonl"
refined_queries_cache_path = '/data/data_public/riverbag/testRepoSummaryOut/Filited/mrjob/refined_queries.json' 

# METHODS_CSV = "/data/data_public/riverbag/testRepoSummaryOut/Filited/boto/methods.csv" 
# FILTERED_FILE = "/data/data_public/riverbag/testRepoSummaryOut/Filited/boto/filtered.jsonl"
# refined_queries_cache_path = '/data/data_public/riverbag/testRepoSummaryOut/Filited/boto/refined_queries.json' 

# METHODS_CSV = "/data/data_public/riverbag/testRepoSummaryOut/Filited/alembic/methods.csv" 
# FILTERED_FILE = "/data/data_public/riverbag/testRepoSummaryOut/Filited/alembic/filtered.jsonl"
# refined_queries_cache_path = '/data/data_public/riverbag/testRepoSummaryOut/Filited/alembic/refined_queries.json' 

# 是否需要把method名称规范化，例如得到的csv中是mrjob.mrjob.xx，将其规范化为mrjob.xx，以便进行测评
NEED_METHOD_NAME_NORM = False
USE_REFINED_QUERY = False

def load_unixcoder_model(model_path_or_name=None, device=None):
	"""
	Attempt to load UniXcoder model and move it to a device.
	Returns (model, device).
	"""
	if device is None:
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	try:
		model = UniXcoder(model_path_or_name if model_path_or_name else "microsoft/unixcoder-base")
		model.to(device)
		model.eval()
		return model, device
	except Exception:
		raise ImportError("Unable to import UniXcoder. Ensure UniXcoder is installed / in PYTHONPATH and adjust load_unixcoder_model().")

def encode_corpus_with_unixcoder(model, device, texts, batch_size=32, max_length=512):
	"""
	Encode a list of texts (function code strings) into normalized embeddings using UniXcoder.
	We tokenize and encode one example at a time (model.tokenize([text])) so source_ids is a rectangular tensor.
	Returns a torch.FloatTensor of shape (N, D) on CPU (normalized).
	"""
	all_embs = []
	for i in range(0, len(texts), batch_size):
		batch = texts[i:i+batch_size]
		for func in batch:
			# tokenize one example as in official snippet to ensure rectangular tensor shape
			tokens_ids = model.tokenize([func], max_length=max_length, mode="<encoder-only>")
			source_ids = torch.tensor(tokens_ids).to(device)  # shape (1, L)
			with torch.no_grad():
				_, func_embedding = model(source_ids)
				normed = F.normalize(func_embedding, p=2, dim=1)  # shape (1, D)
				all_embs.append(normed.cpu())
	# concat
	if len(all_embs) == 0:
		return torch.empty((0,0))
	code_embs = torch.cat(all_embs, dim=0)
	return code_embs  # on CPU

def encode_nl_with_unixcoder(model, device, text, max_length=512):
	# tokenize one NL example and encode
	tokens_ids = model.tokenize([text], max_length=max_length, mode="<encoder-only>")
	source_ids = torch.tensor(tokens_ids).to(device)
	with torch.no_grad():
		_, nl_embedding = model(source_ids)
		nl_embedding = F.normalize(nl_embedding, p=2, dim=1)  # shape (1, dim)
	return nl_embedding.cpu()

def evaluate_retrieval_with_unixcoder(methods_csv=METHODS_CSV, filtered_file=FILTERED_FILE, model_path=None, batch_size=32):
	# load methods
	methods_df = pd.read_csv(methods_csv, dtype=str).fillna('')
	if 'method_code' not in methods_df.columns or 'method_signature' not in methods_df.columns:
		raise ValueError("methods.csv must contain 'method_code' and 'method_signature' columns")
	method_codes = methods_df['method_code'].tolist()
	method_signatures = methods_df['method_signature'].tolist()

	method_names = [sig.split('(')[0] for sig in method_signatures]
	
	if NEED_METHOD_NAME_NORM:
		# 规范化 method_name：去掉参数，只保留第一个点后的部分，例如 mrjob.hadoop.main -> hadoop.main
		base_names = methods_df['method_signature'].astype(str).str.split('(').str[0]
		methods_df['method_name_norm'] = base_names.str.split('.', n=1).str[1].fillna(base_names)
		method_names = methods_df['method_name_norm'].unique().tolist()

	# load model and device
	model, device = load_unixcoder_model(model_path)

	# encode all method codes
	print("Encoding all method_code with UniXcoder (this may take a while)...")
	code_embs = encode_corpus_with_unixcoder(model, device, method_codes, batch_size=batch_size)
	if code_embs.numel() == 0:
		raise RuntimeError("No embeddings produced for method_code corpus.")
	# Put code embeddings on device for fast dot-product with nl embeddings
	code_embs_device = code_embs.to(device)

	# prepare metrics accumulators for top5, top10, top15
	topk_list = [5, 10, 15]
	match_counts = {k: 0 for k in topk_list}
	pred_counts = {k: 0 for k in topk_list}
	total_gt = 0

	# Load or initialize the query cache
	if os.path.exists(refined_queries_cache_path):
		with open(refined_queries_cache_path, 'r') as f:
			refined_queries_cache = json.load(f)
	else:
		refined_queries_cache = {}
		with open(refined_queries_cache_path, 'w') as f:
			json.dump(refined_queries_cache, f, indent=2)

	# read filtered queries
	with open(filtered_file, 'r') as f:
		example_counter = 0
		unixcoder_records = []
		for line in f:
			example_counter += 1
			data = json.loads(line.strip())
			# collect deps
			deps = []
			deps.extend(data['dependency']['intra_class'])
			deps.extend(data['dependency']['intra_file'])
			deps.extend(data['dependency']['cross_file'])
			# filter deps to known method names (method name without args)
			
			deps = [dep for dep in deps if dep in method_names]
			total_gt += len(deps)

			# build natural language query
			#query = data['requirement']['Functionality'] + ' ' + data['requirement']['Arguments']
			original_query = data['requirement']['Functionality'] + ' ' + data['requirement']['Arguments']
			#print("original query: ", original_query)

			if USE_REFINED_QUERY:
				if original_query in refined_queries_cache:
					query = refined_queries_cache[original_query]
					print("found in cache")
				else:
					modelname = "deepseek-v3"
					query = refine_query(original_query, modelname)
					refined_queries_cache[original_query] = query
					# 关键：添加后立即保存！
					with open(refined_queries_cache_path, 'w') as f:
						json.dump(refined_queries_cache, f, indent=2)
				print("refined query: ", query)
			else:
				query = original_query
				print("Using original query: ", query)

			# encode query NL using UniXcoder
			nl_emb = encode_nl_with_unixcoder(model, device, query)  # CPU tensor
			nl_emb = nl_emb.to(device)

			# similarities: dot product between normalized nl_emb (1,dim) and code_embs_device (N,dim)^T => (1,N)
			with torch.no_grad():
				sims = torch.mm(nl_emb, code_embs_device.t()).squeeze(0)  # shape (N,)
			# get topk indices for each k
			unixcoder_record = {
				"example_id": example_counter,
				"query": query,
				"ground_truth": deps,
				"unixcoder_code": {}
			}
			for k in topk_list:
				if sims.numel() == 0:
					topk_idx = np.array([], dtype=int)
				else:
					topk_idx = torch.topk(sims, k=min(k, sims.numel()), largest=True).indices.cpu().numpy()
				# predicted method signatures
				pred_methods = [method_signatures[i] for i in topk_idx]
				num_pred = len(pred_methods)
				num_match = sum(1 for dep in deps if any(dep in m for m in pred_methods))
				num_gt = len(deps)
				precision = (num_match / num_pred) if num_pred > 0 else 0
				recall = (num_match / num_gt) if num_gt > 0 else 0
				f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

				pred_counts[k] += num_pred
				match_counts[k] += num_match

				unixcoder_record["unixcoder_code"][f"top{k}"] = {
					"metrics": {
						"P": precision,
						"R": recall,
						"F1": f1,
						"pred": num_pred,
						"match": num_match,
						"gt": num_gt
					},
					"predictions": [{"method": m, "match": any(dep in m for dep in deps)} for m in pred_methods]
				}
			unixcoder_records.append(unixcoder_record)

	out_dir = os.path.dirname(filtered_file)
	unixcoder_path = os.path.join(out_dir, "diagnostic_unixcoder_code.jsonl")
	with open(unixcoder_path, "w", encoding="utf-8") as uo:
		for rec in unixcoder_records:
			uo.write(json.dumps(rec, ensure_ascii=False) + "\n")

	# print metrics
	def safe_div(a, b):
		return (a / b) if b != 0 else 0

	print("UniXcoder retrieval results (based on method_code embeddings):")
	for k in topk_list:
		m = match_counts[k]
		p = pred_counts[k]
		r = total_gt
		precision = safe_div(m, p)
		recall = safe_div(m, r)
		f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
		print(f"Top{k}: Match={m}, Pred={p}, GT={r}, P={(precision*100):.2f}%, R={(recall*100):.2f}%, F1={(f1*100):.2f}%")

if __name__ == "__main__":
	# adjust these paths / model checkpoint as needed
	methods_csv = METHODS_CSV
	filtered_file = FILTERED_FILE
	# model_path can be None if your load_unixcoder_model defaults to a sensible checkpoint
	model_path = None
	evaluate_retrieval_with_unixcoder(methods_csv=methods_csv, filtered_file=filtered_file, model_path=model_path, batch_size=32)
