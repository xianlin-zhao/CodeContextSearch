import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from typing import Any, Dict, Optional
from search_models.unixcoder import UniXcoder
from utils.query_refine import refine_query

# METHODS_CSV = "/data/zxl/Search2026/outputData/repoSummaryOut/mrjob/1111/methods.csv"
# FILTERED_FILE = "/data/zxl/Search2026/outputData/repoSummaryOut/mrjob/1111/filtered.jsonl"

# METHODS_CSV = "/home/riverbag/testRepoSummaryOut/boto/boto_testAug/1122_codet5/methods.csv" 
# FILTERED_FILE = "/home/riverbag/testRepoSummaryOut/boto/boto_testAug/1122_codet5/filtered.jsonl"

METHODS_CSV = "/data/data_public/riverbag/testRepoSummaryOut/211/diffprivlib/methods.csv" 
FILTERED_FILE = "/data/zxl/Search2026/outputData/devEvalSearchOut/diffprivlib/0228/filtered.jsonl" 
refined_queries_cache_path = '/data/zxl/Search2026/outputData/devEvalSearchOut/diffprivlib/0228/refined_queries.json'
ENRE_JSON = "/data/data_public/riverbag/testRepoSummaryOut/211/diffprivlib/diffprivlib-report-enre.json"

# METHODS_CSV = "/data/data_public/riverbag/testRepoSummaryOut/Filited/boto/methods.csv" 
# FILTERED_FILE = "/data/data_public/riverbag/testRepoSummaryOut/Filited/boto/filtered.jsonl"
# refined_queries_cache_path = '/data/data_public/riverbag/testRepoSummaryOut/Filited/boto/refined_queries.json' 
# ENRE_JSON = "/data/data_public/riverbag/testRepoSummaryOut/Filited/boto/boto-report-enre.json"

# METHODS_CSV = "/data/data_public/riverbag/testRepoSummaryOut/Filited/alembic/methods.csv" 
# FILTERED_FILE = "/data/data_public/riverbag/testRepoSummaryOut/Filited/alembic/filtered.jsonl"
# refined_queries_cache_path = '/data/data_public/riverbag/testRepoSummaryOut/Filited/alembic/refined_queries.json' 
# ENRE_JSON = "/data/data_public/riverbag/testRepoSummaryOut/Filited/alembic/alembic-report-enre.json"

# 是否需要把method名称规范化，例如得到的csv中是mrjob.mrjob.xx，将其规范化为mrjob.xx，以便进行测评
NEED_METHOD_NAME_NORM = False
USE_REFINED_QUERY = False

variables_enre = set()  # 变量类型，只要搜到的代码里用到了这个变量，就认为成功
unresolved_attribute_enre = set()  # enre中的此类型通常表示一个类里的self.xxx属性，只要搜到的代码出现了self.xxx，就认为成功
module_enre = set()  # 模块（其实是python文件），有时候dependency里会出现单独的模块名，只要搜到这个模块里的元素，就认为成功
package_enre = set()  # 包，会出现与module类似的情况


def load_enre_elements(json_path):
	"""读取enre的解析结果文件，重点读取Variable, Unresolved Attribute, Module, Package类型"""
	if not os.path.exists(json_path):
		print(f"Warning: ENRE JSON file not found at {json_path}")
		return

	try:
		with open(json_path, 'r') as f:
			data = json.load(f)
	except Exception as e:
		print(f"Error reading ENRE JSON: {e}")
		return

	variables = data.get("variables", [])
	for var in variables:
		if var.get('category') == 'Variable':
			qname = var.get('qualifiedName')
			if qname:
				variables_enre.add(qname)
		elif var.get('category') == 'Unresolved Attribute':
			# 必须存在"File"字段，且File必须包含"/"，否则可能是外部库文件里面的属性
			if 'File' in var and '/' in var.get('File'):
				qname = var.get('qualifiedName')
				if qname:
					unresolved_attribute_enre.add(qname)
		elif var.get('category') == 'Module':
			qname = var.get('qualifiedName')
			if qname:
				module_enre.add(qname)
		elif var.get('category') == 'Package':
			qname = var.get('qualifiedName')
			if qname:
				package_enre.add(qname)


def _normalize_symbol(s: str) -> str:
	if "(" in s:
		return s.split("(", 1)[0]
	return s


def compute_task_recall(
	dependency: Optional[list[str]],
	searched_context_code_list: list[Dict[str, Any]],
) -> Dict[str, Any]:
	dep = dependency or []
	dep_set = {x for x in dep}

	# 这里搜到的只能是函数或类
	retrieved_set = {
		_normalize_symbol(str(x.get("method_signature", "")))
		for x in searched_context_code_list
		if isinstance(x, dict)
	}
	# 特判：如果retrieve的元素是xx.__init__.yy这种格式，要转成xx.yy，因为enre的解析结果和DevEval不一样
	retrieved_set = {x.replace(".__init__", "") for x in retrieved_set}

	dep_total = len(dep_set)

	hit_set = dep_set & retrieved_set
	hit = len(hit_set) if dep_total > 0 else 0

	# 特判
	for x in dep:
		if x in variables_enre:
			# 如果是变量，就看该变量是否出现在某段代码里
			var_name = x.split('.')[-1]
			for context_code in searched_context_code_list:
				code_detail = context_code.get("method_code", "")
				if var_name in code_detail:
					hit_set.add(x)
					hit += 1
					break
		elif x in unresolved_attribute_enre:
			attr_name = x.split('.')[-1]  # 属性名
			class_name = '.'.join(x.split('.')[:-1])  # 类名是属性名前面的全部
			for context_code in searched_context_code_list:
				sig = context_code.get("sig", "")
				code_detail = context_code.get("method_code", "")
				# 如果这段搜到的代码真的在class_name类里面，且包含self.xxx属性，就认为成功
				if sig.startswith(f"{class_name}.") and f"self.{attr_name}" in code_detail:
					hit_set.add(x)
					hit += 1
					break
		elif x in module_enre:
			# 如果是模块，就看该模块名称是否为某个context_code的sig的前缀
			module_name = x
			for context_code in searched_context_code_list:
				sig = context_code.get("sig", "")
				if sig.startswith(module_name):
					hit_set.add(x)
					hit += 1
					break
		elif x in package_enre:
			# 如果是包，就看该包名称是否为某个context_code的sig的前缀，与Module类似
			package_name = x
			for context_code in searched_context_code_list:
				sig = context_code.get("sig", "")
				if sig.startswith(package_name):
					hit_set.add(x)
					hit += 1
					break

	recall = (hit / dep_total) if dep_total > 0 else None
	return {
		"dependency_total": dep_total,
		"dependency_hit": hit,
		"recall": recall,
	}


def is_method_hit(method_signature: str, method_code: str, deps_list: list[str]) -> bool:
	norm_sig = _normalize_symbol(method_signature).replace(".__init__", "")
	for x in deps_list:
		if x == norm_sig:
			return True
		if x in variables_enre:
			var_name = x.split(".")[-1]
			if var_name and var_name in (method_code or ""):
				return True
		if x in unresolved_attribute_enre:
			attr_name = x.split('.')[-1]
			class_name = '.'.join(x.split('.')[:-1])
			if norm_sig.startswith(f"{class_name}.") and f"self.{attr_name}" in (method_code or ""):
				return True
		if x in module_enre:
			if norm_sig.startswith(x):
				return True
		if x in package_enre:
			if norm_sig.startswith(x):
				return True
	return False


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

	load_enre_elements(ENRE_JSON)

	method_sig_to_code = dict(
        zip(
            methods_df["method_signature"].astype(str).tolist(),
            methods_df["method_code"].astype(str).tolist(),
        )
    )

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
			
			# deps = [dep for dep in deps if (dep in method_names) or (dep in variables_enre)]
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
				searched_context_code_list = [
					{
						"sig": _normalize_symbol(method_signatures[i]),
						"method_signature": method_signatures[i],
						"method_code": method_codes[i],
					}
					for i in topk_idx
				]
				num_pred = len(pred_methods)
				recall_info = compute_task_recall(deps, searched_context_code_list)
				num_match = int(recall_info["dependency_hit"])
				num_gt = int(recall_info["dependency_total"])
				precision = (num_match / num_pred) if num_pred > 0 else 0
				recall = float(recall_info["recall"]) if recall_info["recall"] is not None else 0
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
					"predictions": [
						{
							"method": m,
							"match": is_method_hit(m, method_sig_to_code.get(m, ""), deps)
						}
						for m in pred_methods
					]
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
