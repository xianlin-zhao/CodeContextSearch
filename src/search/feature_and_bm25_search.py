import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
import json
import pandas as pd
from typing import Any, Dict, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from rank_bm25 import BM25Okapi
from utils.query_refine import refine_query

# PROJECT_PATH = "System/mrjob"
# PROJECT_PATH = "Internet/boto"
# PROJECT_PATH ="Database/alembic"
# PROJECT_PATH = "Multimedia/Mopidy"
PROJECT_PATH = "Security/diffprivlib"

# FEATURE_CSV = "/data/data_public/riverbag/testRepoSummaryOut/Filited/mrjob/features.csv" 
# METHODS_CSV = "/data/data_public/riverbag/testRepoSummaryOut/Filited/mrjob/methods.csv" 
# METHODS_DESC_CSV = "/data/data_public/riverbag/testRepoSummaryOut/Filited/mrjob/methods_with_desc.csv"
# FILTERED_PATH = "/data/data_public/riverbag/testRepoSummaryOut/Filited/mrjob/filtered.jsonl" 
# refined_queries_cache_path = '/data/data_public/riverbag/testRepoSummaryOut/Filited/mrjob/refined_queries.json' 

# FEATURE_CSV = "/data/data_public/riverbag/testRepoSummaryOut/Filited/boto/features.csv" 
# METHODS_CSV = "/data/data_public/riverbag/testRepoSummaryOut/Filited/boto/methods.csv" 
# METHODS_DESC_CSV = "/data/data_public/riverbag/testRepoSummaryOut/Filited/boto/methods_with_desc.csv"
# FILTERED_PATH = "/data/data_public/riverbag/testRepoSummaryOut/Filited/boto/filtered.jsonl"
# refined_queries_cache_path = '/data/data_public/riverbag/testRepoSummaryOut/Filited/boto/refined_queries.json' 
# ENRE_JSON = "/data/data_public/riverbag/testRepoSummaryOut/Filited/boto/boto-report-enre.json"

FEATURE_CSV = "/data/data_public/riverbag/testRepoSummaryOut/211/diffprivlib/features.csv" 
METHODS_CSV = "/data/data_public/riverbag/testRepoSummaryOut/211/diffprivlib/methods.csv" 
METHODS_DESC_CSV = "/data/data_public/riverbag/testRepoSummaryOut/211/diffprivlib/methods_with_desc.csv"
FILTERED_PATH = "/data/zxl/Search2026/outputData/devEvalSearchOut/diffprivlib/0228/filtered.jsonl" 
refined_queries_cache_path = '/data/zxl/Search2026/outputData/devEvalSearchOut/diffprivlib/0228/refined_queries.json' 
ENRE_JSON = "/data/data_public/riverbag/testRepoSummaryOut/211/diffprivlib/diffprivlib-report-enre.json"
# DevEval数据集case的路径（json，不是数据集项目本身）
DATA_JSONL = "/data/lowcode_public/DevEval/data_have_dependency_cross_file.jsonl"


# 是否需要把method名称规范化，例如得到的csv中是mrjob.mrjob.xx，将其规范化为mrjob.xx，以便进行测评
NEED_METHOD_NAME_NORM = False
USE_REFINED_QUERY = False
TOP_KS = [1, 2, 3]

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

def analyze_project(project_path):
    # Create output directory
    # output_dir = project_path
    # os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Filter by project_path
    with open(DATA_JSONL, 'r') as infile, open(FILTERED_PATH, 'w') as outfile:
        for line in infile:
            data = json.loads(line.strip())
            if data.get('project_path') == project_path:
                outfile.write(line)
    
    # Step 2: Find similar clusters
    df = pd.read_csv(
        FEATURE_CSV,
        dtype=str,
        keep_default_na=False,
        quoting=0,
        engine="python",
        on_bad_lines="skip"
    )
    clusters = df.groupby('id')['desc'].first().reset_index()
    # 提取clusters中所有的method_name
    method_names = df['method_name'].str.split('(').str[0].unique().tolist()

    if NEED_METHOD_NAME_NORM:
        # 规范化 method_name：去掉参数，只保留第一个点后的部分，例如 mrjob.hadoop.main -> hadoop.main
        base_names = df['method_name'].astype(str).str.split('(').str[0]
        df['method_name_norm'] = base_names.str.split('.', n=1).str[1].fillna(base_names)
        method_names = df['method_name_norm'].unique().tolist()
    #print(method_names[:30])

    method_norm_to_feature_id = {}
    for fid, m in zip(df["id"].astype(str).tolist(), df["method_name"].astype(str).tolist()):
        m_norm = _normalize_symbol(m)
        if m_norm and m_norm not in method_norm_to_feature_id:
            method_norm_to_feature_id[m_norm] = fid

    # #model = SentenceTransformer('all-mpnet-base-v2')
    # model = SentenceTransformer('all-MiniLM-L6-v2')
    # 检查路径是否存在
    model_path = "/data/data_public/riverbag/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots/c9745ed1d9f207416be6d2e6f8de32d1f16199bf"

    if os.path.exists(model_path):
        print(f"找到本地模型: {model_path}")
        model = SentenceTransformer(model_path)
    else:
        print(f"未找到本地模型，尝试下载...")
        # 如果本地没有，再尝试从网络下载
        model = SentenceTransformer('all-MiniLM-L6-v2')
    cluster_embeddings = model.encode(clusters['desc'].tolist())

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
    print(signature_docs[:10])
    print("=======================================")
    code_docs = [tokenize_code(c) for c in methods_df['method_code'].tolist()]
    print(code_docs[:10])
    print("=======================================")
    desc_docs = [tokenize_text(d) for d in methods_desc_df['func_desc'].tolist()]
    print(desc_docs[:10])
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
        feature_records = []
        sig_records = []
        code_records = []
        desc_records = []
        top_gt = 0

        cluster_ks = [1, 3, 5]
        cluster_metrics = {k: {"match": 0, "pred": 0} for k in cluster_ks}

        sig_ks = [5, 10, 15]
        sig_metrics = {k: {"match": 0, "pred": 0} for k in sig_ks}

        code_metrics = {k: {"match": 0, "pred": 0} for k in sig_ks}
        desc_metrics = {k: {"match": 0, "pred": 0} for k in sig_ks}

        def methods_from_top_k_clusters(similarities, k):
            idx = np.argsort(similarities)[-k:][::-1]
            ids = clusters.iloc[idx]["id"].tolist()
            methods = []
            for cid in ids:
                methods.extend(df[df["id"] == cid]["method_name"].tolist())
            return methods

        def bm25_topk_strings(order, k, corpus_strings):
            idx = order[-k:][::-1]
            return [corpus_strings[i] for i in idx]

        for line in f:
            example_counter += 1
            data = json.loads(line.strip())
            deps = []
            target_method = data.get("namespace") or ""
            target_feature_id = method_norm_to_feature_id.get(target_method)
            target_feature_other_methods = []
            if target_feature_id is not None:
                target_feature_methods = (
                    df.loc[df["id"].astype(str) == str(target_feature_id), "method_name"]
                    .astype(str)
                    .tolist()
                )
                target_method_norm = _normalize_symbol(target_method)
                target_feature_other_methods = [
                    _normalize_symbol(m) for m in target_feature_methods if _normalize_symbol(m) != target_method_norm
                ]

            target_method_sig = resolve_method_signature(target_method)
            target_method_code = method_sig_to_code.get(target_method_sig, "")
            similar_methods = {}
            target_code_tokens = tokenize_code(target_method_code) if target_method_code else []
            if target_code_tokens:
                target_code_scores = bm25_code.get_scores(target_code_tokens)
                target_code_order = np.argsort(target_code_scores)
                target_method_norm = _normalize_symbol(target_method)
                for k in TOP_KS:
                    k_int = int(k)
                    selected = []
                    seen = set()
                    for idx in target_code_order[::-1]:
                        m = methods_corpus_strings[int(idx)]
                        m_norm = _normalize_symbol(m)
                        if m_norm == target_method_norm:
                            continue
                        if m_norm in seen:
                            continue
                        selected.append(m_norm)
                        seen.add(m_norm)
                        if len(selected) >= k_int:
                            break
                    similar_methods[f"top{k_int}"] = selected
            else:
                for k in TOP_KS:
                    similar_methods[f"top{int(k)}"] = []
            #从数据中提取真实的依赖关系（ dependency ），这些是本次搜索的“正确答案”
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

            # feature-based search
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
            #input("please confirm the query!")
            query_embedding = model.encode([query])
            similarities = cosine_similarity(query_embedding, cluster_embeddings)[0]

            for k in cluster_ks:
                methods_k = methods_from_top_k_clusters(similarities, k)
                cluster_metrics[k]["pred"] += len(methods_k)
                cluster_metrics[k]["match"] += compute_task_recall(deps, build_context_code_list(methods_k))["dependency_hit"]

            # signature-based search using BM25
            q_tokens = re.findall(r'\w+', query.lower())
            sig_scores = bm25_sig.get_scores(q_tokens)
            sig_order = np.argsort(sig_scores)
            for k in sig_ks:
                m = bm25_topk_strings(sig_order, k, methods_corpus_strings)
                sig_metrics[k]["pred"] += len(m)
                sig_metrics[k]["match"] += compute_task_recall(deps, build_context_code_list(m))["dependency_hit"]

            # code-based search using BM25
            code_scores = bm25_code.get_scores(q_tokens)
            code_order = np.argsort(code_scores)
            for k in sig_ks:
                m = bm25_topk_strings(code_order, k, methods_corpus_strings)
                code_metrics[k]["pred"] += len(m)
                code_metrics[k]["match"] += compute_task_recall(deps, build_context_code_list(m))["dependency_hit"]

            # desc-based search using BM25
            desc_scores = bm25_desc.get_scores(q_tokens)
            desc_order = np.argsort(desc_scores)
            for k in sig_ks:
                m = bm25_topk_strings(desc_order, k, methods_desc_corpus_strings)
                desc_metrics[k]["pred"] += len(m)
                desc_metrics[k]["match"] += compute_task_recall(deps, build_context_code_list(m))["dependency_hit"]

            feature_record = {
                "example_id": example_counter,
                "query": query,
                "target_method": target_method,
                "similar_methods": similar_methods,
                "target_feature_id": target_feature_id,
                "target_feature_other_methods": target_feature_other_methods,
                "ground_truth": deps,
                "feature": {}
            }
            for k in cluster_ks:
                mk = methods_from_top_k_clusters(similarities, k)
                num_pred = len(mk)
                recall_info = compute_task_recall(deps, build_context_code_list(mk))
                num_match = int(recall_info["dependency_hit"])
                num_gt = int(recall_info["dependency_total"])
                precision = (num_match / num_pred) if num_pred > 0 else 0
                recall = float(recall_info["recall"]) if recall_info["recall"] is not None else 0
                f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                feature_record["feature"][f"top{k}"] = {
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
            feature_records.append(feature_record)

            sig_record = {
                "example_id": example_counter,
                "query": query,
                "ground_truth": deps,
                "bm25_signature": {}
            }
            for k in sig_ks:
                mk = bm25_topk_strings(sig_order, k,methods_corpus_strings)
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
        feature_path = os.path.join(out_dir, "diagnostic_feature.jsonl")
        sig_path = os.path.join(out_dir, "diagnostic_bm25_signature.jsonl")
        code_path = os.path.join(out_dir, "diagnostic_bm25_code.jsonl")
        desc_path = os.path.join(out_dir, "diagnostic_bm25_desc.jsonl")
        with open(feature_path, "w", encoding="utf-8") as fo:
            for rec in feature_records:
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
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

        for k in cluster_ks:
            m = cluster_metrics[k]["match"]
            p = cluster_metrics[k]["pred"]
            print(f"Top {k} Match: {m}")
            print(f"Top {k} Pred: {p}")
            print(f"Top {k} P={(safe_div(m, p))*100:.2f}%")
            print(f"Top {k} R={(safe_div(m, top_gt))*100:.2f}%")
            denom = safe_div(m, p) + safe_div(m, top_gt)
            f1_val = (2 * safe_div(m, p) * safe_div(m, top_gt) / denom) if denom > 0 else 0
            print(f"Top {k} F1={f1_val*100:.2f}%")
            print("--------------------------------")

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
            print(f"Top{k} Match: {m}, Pred: {p}, P={(safe_div(m, p))*100:.2f}%, R={(safe_div(m, top_gt))*100:.2f}%, F1={f1_val*100:.2f}%")
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
    
    print(f"Analysis completed for {project_path}")


if __name__ == "__main__":
    analyze_project(PROJECT_PATH)
