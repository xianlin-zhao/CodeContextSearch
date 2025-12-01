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

METHODS_CSV = "/home/riverbag/testRepoSummaryOut/boto/boto_testAug/1122_codet5/methods.csv" 
FILTERED_FILE = "/home/riverbag/testRepoSummaryOut/boto/boto_testAug/1122_codet5/filtered.jsonl"

# METHODS_CSV = "/home/riverbag/testRepoSummaryOut/mrjob/1122_codet5/methods.csv" 
# FILTERED_FILE = "/home/riverbag/testRepoSummaryOut/mrjob/1122_codet5/filtered.jsonl"

# 是否需要把method名称规范化，例如得到的csv中是mrjob.mrjob.xx，将其规范化为mrjob.xx，以便进行测评
NEED_METHOD_NAME_NORM = True

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

	# read filtered queries
	with open(filtered_file, 'r') as f:
		for line in f:
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
			query = data['requirement']['Functionality'] + ' ' + data['requirement']['Arguments']
			# encode query NL using UniXcoder
			nl_emb = encode_nl_with_unixcoder(model, device, query)  # CPU tensor
			nl_emb = nl_emb.to(device)

			# similarities: dot product between normalized nl_emb (1,dim) and code_embs_device (N,dim)^T => (1,N)
			with torch.no_grad():
				sims = torch.mm(nl_emb, code_embs_device.t()).squeeze(0)  # shape (N,)
			# get topk indices for each k
			for k in topk_list:
				if sims.numel() == 0:
					topk_idx = np.array([], dtype=int)
				else:
					topk_idx = torch.topk(sims, k=min(k, sims.numel()), largest=True).indices.cpu().numpy()
				# predicted method signatures
				pred_methods = [method_signatures[i] for i in topk_idx]
				pred_counts[k] += len(pred_methods)
				# matching: if a dependency name appears in the predicted signature string
				for dep in deps:
					for method in pred_methods:
						if dep in method:
							match_counts[k] += 1
							break

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