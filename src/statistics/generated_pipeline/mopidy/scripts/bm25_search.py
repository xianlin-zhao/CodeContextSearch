import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
import json
import pandas as pd
from typing import Any, Dict, Optional
import numpy as np
import re
from rank_bm25 import BM25Okapi
from utils.query_refine import refine_query

PROJECT_DIR = "Multimedia/Mopidy"
# PROJECT_DIR = "Internet/boto"
# PROJECT_DIR ="Database/alembic"
# PROJECT_DIR = "Multimedia/Mopidy"
# PROJECT_DIR = "Security/diffprivlib"
# PROJECT_DIR = "Security/diffprivlib"

METHODS_CSV = "/data/data_public/riverbag/testRepoSummaryOut/211/mopidy/methods.csv"
METHODS_DESC_CSV = "/data/data_public/riverbag/testRepoSummaryOut/211/mopidy/methods_with_desc.csv"
FILTERED_PATH = "/data/data_public/riverbag/testRepoSummaryOut/211/mopidy/filtered.jsonl"
refined_queries_cache_path = "/data/data_public/riverbag/testRepoSummaryOut/211/mopidy/refined_queries.json"
ENRE_JSON = "/data/data_public/riverbag/testRepoSummaryOut/211/mopidy/mopidy-report-enre.json"
# DevEval数据集case的路径（json，不是数据集项目本身）
# DATA_JSONL = "/data/lowcode_public/DevEval/data_have_dependency_cross_file.jsonl"
# 完整数据集jsonl路径
DATA_JSONL = "/data/zxl/Search2026/DevEval/data.jsonl"


# 是否需要把method名称规范化，例如得到的csv中是mrjob.mrjob.xx，将其规范化为mrjob.xx，以便进行测评
NEED_METHOD_NAME_NORM = False
USE_REFINED_QUERY = False


variables_enre = set()  # 变量类型，只要搜到的代码里用到了这个变量，就认为成功
unresolved_attribute_enre = set()  # enre中的此类型通常表示一个类里的self.xxx属性，只要搜到的代码出现了self.xxx，就认为成功
module_enre = set()  # 模块(其实是python文件)，有时候dependency里会出现单独的模块名，只要搜到这个模块里的元素，就认为成功
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


def _maybe_insert_init_in_target_method(target_method: str) -> str:
    if not target_method:
        return target_method
    base = target_method.split("(", 1)[0] if "(" in target_method else target_method
    if ".__init__." in base:
        return base
    parts = [p for p in base.split(".") if p]
    if len(parts) < 2:
        return base
    insert_at = None
    for i, part in enumerate(parts):
        if any(c.isupper() for c in part):
            insert_at = i
            break
    if insert_at is None:
        insert_at = len(parts) - 1
    insert_at = max(1, insert_at)
    parts.insert(insert_at, "__init__")
    return ".".join(parts)


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

def analyze_project(project_dir):
    # Create output directory
    # output_dir = project_dir
    # os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Filter by project_dir
    with open(DATA_JSONL, 'r') as infile, open(FILTERED_PATH, 'w') as outfile:
        for line in infile:
            data = json.loads(line.strip())
            if data.get('project_path') == project_dir:
                outfile.write(line)
    
    # load methods corpus
    methods_df = pd.read_csv(METHODS_CSV, dtype=str).fillna('')
    # ensure columns exist
    if 'method_signature' not in methods_df.columns or 'method_code' not in methods_df.columns:
        raise ValueError("methods.csv must contain 'method_signature' and 'method_code' columns")

    # load methods with description corpus
    methods_desc_df = pd.read_csv(METHODS_DESC_CSV, dtype=str).fillna('')
    if 'func_desc' not in methods_desc_df.columns:
        raise ValueError("methods_with_desc.csv must contain 'func_desc' column")

    def tokenize_signature(sig: str):
        # keep part after last '.', remove punctuation, replace underscores with spaces, split into tokens
        part = sig.split('.')[-1]
        part = part.replace('_', ' ')
        part = re.sub(r'[\(\),.:]', ' ', part)
        toks = re.findall(r'\w+', part.lower())
        return toks

    def tokenize_code(code: str):
        toks = re.findall(r'\w+', code.lower())
        return toks

    def tokenize_text(text: str):
        toks = re.findall(r'\w+', text.lower())
        return toks

    signature_docs = [tokenize_signature(s) for s in methods_df['method_signature'].tolist()]
    # print(signature_docs[:10])
    # print("=======================================")
    code_docs = [tokenize_code(c) for c in methods_df['method_code'].tolist()]
    # print(code_docs[:10])
    # print("=======================================")
    desc_docs = [tokenize_text(d) for d in methods_desc_df['func_desc'].tolist()]
    # print(desc_docs[:10])
    # store the original method strings to return as predicted items
    methods_corpus_strings = methods_df['method_signature'].tolist()
    # 这里其实methods_desc_df['func_fullName']和methods_df['method_signature']是一样的，但是为了防止不一样的情况，这里还是给他进行了单独拎出来
    methods_desc_corpus_strings = methods_desc_df['func_fullName'].tolist()

    bm25_sig = BM25Okapi(signature_docs)
    bm25_code = BM25Okapi(code_docs)
    bm25_desc = BM25Okapi(desc_docs)

    load_enre_elements(ENRE_JSON)

    method_sig_to_code = dict(
        zip(
            methods_df["method_signature"].astype(str).tolist(),
            methods_df["method_code"].astype(str).tolist(),
        )
    )
    normalized_sig_to_signature = {}
    for sig in method_sig_to_code.keys():
        norm = _normalize_symbol(sig)
        if norm not in normalized_sig_to_signature:
            normalized_sig_to_signature[norm] = sig

    def resolve_method_signature(method_str: str) -> str:
        if method_str in method_sig_to_code:
            return method_str
        norm = _normalize_symbol(method_str)
        if norm in normalized_sig_to_signature:
            return normalized_sig_to_signature[norm]
        return method_str

    def build_context_code_list(method_strs: list[str]) -> list[Dict[str, Any]]:
        ctx = []
        for m in method_strs:
            sig = resolve_method_signature(m)
            ctx.append(
                {
                    "sig": _normalize_symbol(sig),
                    "method_signature": sig,
                    "method_code": method_sig_to_code.get(sig, ""),
                }
            )
        return ctx

    def filter_out_target_method(method_strs: list[str], target_method_str: str) -> list[str]:
        target_norm = _normalize_symbol(target_method_str)
        return [m for m in method_strs if _normalize_symbol(m) != target_norm]

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

    # Load or initialize the query cache
    if os.path.exists(refined_queries_cache_path):
        with open(refined_queries_cache_path, 'r') as f:
            refined_queries_cache = json.load(f)
    else:
        refined_queries_cache = {}
        with open(refined_queries_cache_path, 'w') as f:
            json.dump(refined_queries_cache, f, indent=2)

    with open(FILTERED_PATH, 'r') as f:
        example_counter = 0
        sig_records = []
        code_records = []
        desc_records = []
        top_gt = 0

        sig_ks = [5, 10, 15 ,20]
        sig_metrics = {k: {"match": 0, "pred": 0} for k in sig_ks}

        code_metrics = {k: {"match": 0, "pred": 0} for k in sig_ks}
        desc_metrics = {k: {"match": 0, "pred": 0} for k in sig_ks}

        def bm25_topk_strings(order, k, corpus_strings):
            idx = order[-k:][::-1]
            return [corpus_strings[i] for i in idx]

        for line in f:
            example_counter += 1
            data = json.loads(line.strip())
            deps = []
            target_method = data.get("namespace") or ""
            #从数据中提取真实的依赖关系( dependency )，这些是本次搜索的“正确答案”
            deps.extend(data['dependency']['intra_class'])
            deps.extend(data['dependency']['intra_file'])
            deps.extend(data['dependency']['cross_file'])
            # print("deps",deps)
            # #清洗/过滤
            # deps = [dep for dep in deps if (dep in method_names) or (dep in variables_enre)]
            # print("deps after filter",deps)
            # input("please confirm the deps!")
            #将本条测试数据的正确答案数量累加到总数中
            top_gt += len(deps)

            original_query = data['requirement']['Functionality'] + ' ' + data['requirement']['Arguments']
            #print("original query: ", original_query)

            if USE_REFINED_QUERY:
                if original_query in refined_queries_cache:
                    query = refined_queries_cache[original_query]
                    # print("found in cache")
                else:
                    modelname = "deepseek-v3"
                    query = refine_query(original_query, modelname)
                    refined_queries_cache[original_query] = query
                    # 关键：添加后立即保存！
                    with open(refined_queries_cache_path, 'w') as f:
                        json.dump(refined_queries_cache, f, indent=2)
                # print("refined query: ", query)
            else:
                query = original_query
                # print("Using original query: ", query)

            # signature-based search using BM25
            q_tokens = re.findall(r'\w+', query.lower())
            sig_scores = bm25_sig.get_scores(q_tokens)
            sig_order = np.argsort(sig_scores)
            for k in sig_ks:
                m = bm25_topk_strings(sig_order, k, methods_corpus_strings)
                m = filter_out_target_method(m, target_method)
                sig_metrics[k]["pred"] += len(m)
                sig_metrics[k]["match"] += compute_task_recall(deps, build_context_code_list(m))["dependency_hit"]

            # code-based search using BM25
            code_scores = bm25_code.get_scores(q_tokens)
            code_order = np.argsort(code_scores)
            for k in sig_ks:
                m = bm25_topk_strings(code_order, k, methods_corpus_strings)
                m = filter_out_target_method(m, target_method)
                code_metrics[k]["pred"] += len(m)
                code_metrics[k]["match"] += compute_task_recall(deps, build_context_code_list(m))["dependency_hit"]

            # desc-based search using BM25
            desc_scores = bm25_desc.get_scores(q_tokens)
            desc_order = np.argsort(desc_scores)
            for k in sig_ks:
                m = bm25_topk_strings(desc_order, k, methods_desc_corpus_strings)
                m = filter_out_target_method(m, target_method)
                desc_metrics[k]["pred"] += len(m)
                desc_metrics[k]["match"] += compute_task_recall(deps, build_context_code_list(m))["dependency_hit"]

            sig_record = {
                "example_id": example_counter,
                "query": query,
                "ground_truth": deps,
                "bm25_signature": {}
            }
            for k in sig_ks:
                mk = bm25_topk_strings(sig_order, k,methods_corpus_strings)
                mk = filter_out_target_method(mk, target_method)
                num_pred = len(mk)
                recall_info = compute_task_recall(deps, build_context_code_list(mk))
                num_match = int(recall_info["dependency_hit"])
                num_gt = int(recall_info["dependency_total"])
                precision = (num_match / num_pred) if num_pred > 0 else 0
                recall = float(recall_info["recall"]) if recall_info["recall"] is not None else 0
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                sig_record["bm25_signature"][f"top{k}"] = {
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
                            "match": is_method_hit(m, method_sig_to_code.get(m, ""), deps),
                        }
                        for m in mk
                    ]
                }
            sig_records.append(sig_record)

            code_record = {
                "example_id": example_counter,
                "query": query,
                "ground_truth": deps,
                "bm25_code": {}
            }
            for k in sig_ks:
                mk = bm25_topk_strings(code_order, k, methods_corpus_strings)
                mk = filter_out_target_method(mk, target_method)
                num_pred = len(mk)
                recall_info = compute_task_recall(deps, build_context_code_list(mk))
                num_match = int(recall_info["dependency_hit"])
                num_gt = int(recall_info["dependency_total"])
                precision = (num_match / num_pred) if num_pred > 0 else 0
                recall = float(recall_info["recall"]) if recall_info["recall"] is not None else 0
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                code_record["bm25_code"][f"top{k}"] = {
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
                            "match": is_method_hit(m, method_sig_to_code.get(m, ""), deps),
                        }
                        for m in mk
                    ]
                }
            code_records.append(code_record)

            desc_record = {
                "example_id": example_counter,
                "query": query,
                "ground_truth": deps,
                "bm25_desc": {}
            }
            for k in sig_ks:
                mk = bm25_topk_strings(desc_order, k, methods_desc_corpus_strings)
                mk = filter_out_target_method(mk, target_method)
                num_pred = len(mk)
                recall_info = compute_task_recall(deps, build_context_code_list(mk))
                num_match = int(recall_info["dependency_hit"])
                num_gt = int(recall_info["dependency_total"])
                precision = (num_match / num_pred) if num_pred > 0 else 0
                recall = float(recall_info["recall"]) if recall_info["recall"] is not None else 0
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                desc_record["bm25_desc"][f"top{k}"] = {
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
                            "match": is_method_hit(
                                resolve_method_signature(m),
                                method_sig_to_code.get(resolve_method_signature(m), ""),
                                deps,
                            ),
                        }
                        for m in mk
                    ]
                }
            desc_records.append(desc_record)

        out_dir = os.path.dirname(FILTERED_PATH)
        sig_path = os.path.join(out_dir, "diagnostic_bm25_signature.jsonl")
        code_path = os.path.join(out_dir, "diagnostic_bm25_code.jsonl")
        desc_path = os.path.join(out_dir, "diagnostic_bm25_desc.jsonl")
        with open(sig_path, "w", encoding="utf-8") as so:
            for rec in sig_records:
                so.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with open(code_path, "w", encoding="utf-8") as co:
            for rec in code_records:
                co.write(json.dumps(rec, ensure_ascii=False) + "\n")
        with open(desc_path, "w", encoding="utf-8") as do:
            for rec in desc_records:
                do.write(json.dumps(rec, ensure_ascii=False) + "\n")
        
        # Save the updated cache
        with open(refined_queries_cache_path, 'w') as f:
            json.dump(refined_queries_cache, f, indent=4)

        # ===================== DIAGNOSTIC JSON CODE END =====================
        def safe_div(a, b):
            return (a / b) if b != 0 else 0

        # ---- print BM25 signature-based metrics ----
        print("BM25 (signature) results:")
        for k in sig_ks:
            m = sig_metrics[k]["match"]
            p = sig_metrics[k]["pred"]
            denom = safe_div(m, p) + safe_div(m, top_gt)
            f1_val = (2 * safe_div(m, p) * safe_div(m, top_gt) / denom) if denom > 0 else 0
            print(f"Top{k} Match: {m}, Pred: {p}, P={(safe_div(m, p))*100:.2f}%, R={(safe_div(m, top_gt))*100:.2f}%, F1={f1_val*100:.2f}%")
        print("--------------------------------")

        # ---- print BM25 code-based metrics ----
        print("BM25 (code) results:")
        for k in sig_ks:
            m = code_metrics[k]["match"]
            p = code_metrics[k]["pred"]
            #print(f"Top{k} Match: {m}, Pred: {p}, P={(safe_div(m, p))*100:.2f}%, R={(safe_div(m, top_gt))*100:.2f}%")
            denom = safe_div(m, p) + safe_div(m, top_gt)
            f1_val = (2 * safe_div(m, p) * safe_div(m, top_gt) / denom) if denom > 0 else 0
            print(f"Top{k} Match: {m}, Pred: {p}, P={(safe_div(m, p))*100:.2f}%, R={(safe_div(m, top_gt))*100:.2f}%, F1={f1_val*100:.2f}%，{(safe_div(m, top_gt))*100:.2f}%({m}/{p})")
        print("--------------------------------")

        # ---- print BM25 desc-based metrics ----
        print("BM25 (desc) results:")
        for k in sig_ks:
            m = desc_metrics[k]["match"]
            p = desc_metrics[k]["pred"]
            denom = safe_div(m, p) + safe_div(m, top_gt)
            f1_val = (2 * safe_div(m, p) * safe_div(m, top_gt) / denom) if denom > 0 else 0
            print(f"Top{k} Match: {m}, Pred: {p}, P={(safe_div(m, p))*100:.2f}%, R={(safe_div(m, top_gt))*100:.2f}%, F1={f1_val*100:.2f}%")
        print("--------------------------------")
    
    print(f"Analysis completed for {project_dir}")


if __name__ == "__main__":
    analyze_project(PROJECT_DIR)
