import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
import json
import pandas as pd
from typing import Any, Dict, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from rank_bm25 import BM25Okapi
from utils.query_refine import refine_query

PROJECT_DIR = "System/mrjob"
# PROJECT_DIR = "Internet/boto"
# PROJECT_DIR ="Database/alembic"
# PROJECT_DIR = "Multimedia/Mopidy"
# PROJECT_DIR = "Security/diffprivlib"

FEATURE_CSV = "/data/data_public/riverbag/testRepoSummaryOut/211/mrjob/features.csv"
METHODS_CSV = "/data/data_public/riverbag/testRepoSummaryOut/211/mrjob/methods.csv"
FILTERED_PATH = "/data/data_public/riverbag/testRepoSummaryOut/211/mrjob/filtered.jsonl"
refined_queries_cache_path = "/data/data_public/riverbag/testRepoSummaryOut/211/mrjob/refined_queries.json"
ENRE_JSON = "/data/data_public/riverbag/testRepoSummaryOut/211/mrjob/mrjob-report-enre.json"
# DevEval数据集case的路径（json，不是数据集项目本身）
# DATA_JSONL = "/data/lowcode_public/DevEval/data_have_dependency_cross_file.jsonl"
DATA_JSONL = "/data/zxl/Search2026/DevEval/data.jsonl"
TOP_SM = [1, 2, 3]#控制相似的methods的数量
CLUSTER_KS = [1, 3]

# 是否需要把method名称规范化，例如得到的csv中是mrjob.mrjob.xx，将其规范化为mrjob.xx，以便进行测评
NEED_METHOD_NAME_NORM = False
USE_REFINED_QUERY = False

variables_enre = set()
unresolved_attribute_enre = set()
module_enre = set()
package_enre = set()


def load_enre_elements(json_path):
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

    retrieved_set = {
        _normalize_symbol(str(x.get("method_signature", "")))
        for x in searched_context_code_list
        if isinstance(x, dict)
    }
    retrieved_set = {x.replace(".__init__", "") for x in retrieved_set}

    dep_total = len(dep_set)
    hit_set = dep_set & retrieved_set
    hit = len(hit_set) if dep_total > 0 else 0

    for x in dep:
        if x in variables_enre:
            var_name = x.split('.')[-1]
            for context_code in searched_context_code_list:
                code_detail = context_code.get("method_code", "")
                if var_name in code_detail:
                    hit_set.add(x)
                    hit += 1
                    break
        elif x in unresolved_attribute_enre:
            attr_name = x.split('.')[-1]
            class_name = '.'.join(x.split('.')[:-1])
            for context_code in searched_context_code_list:
                sig = context_code.get("sig", "")
                code_detail = context_code.get("method_code", "")
                if sig.startswith(f"{class_name}.") and f"self.{attr_name}" in code_detail:
                    hit_set.add(x)
                    hit += 1
                    break
        elif x in module_enre:
            module_name = x
            for context_code in searched_context_code_list:
                sig = context_code.get("sig", "")
                if sig.startswith(module_name):
                    hit_set.add(x)
                    hit += 1
                    break
        elif x in package_enre:
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

    def tokenize_code(code: str):
        toks = re.findall(r'\w+', code.lower())
        return toks

    code_docs = [tokenize_code(c) for c in methods_df['method_code'].tolist()]
    methods_corpus_strings = methods_df['method_signature'].tolist()
    bm25_code = BM25Okapi(code_docs)

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
        feature_records = []
        top_gt = 0

        cluster_metrics = {k: {"match": 0, "pred": 0} for k in CLUSTER_KS}

        def methods_from_top_k_clusters(similarities, k, forced_cluster_id=None):
            k_int = int(k)
            if k_int == 1 and (forced_cluster_id is None or str(forced_cluster_id).strip() == ""):
                return []
            ranked_idx = np.argsort(similarities)[::-1]
            ranked_ids = clusters.iloc[ranked_idx]["id"].astype(str).tolist()

            selected_ids = []
            if forced_cluster_id is not None:
                forced_id = str(forced_cluster_id)
                if forced_id in ranked_ids:
                    selected_ids.append(forced_id)

            for cid in ranked_ids:
                if cid in selected_ids:
                    continue
                selected_ids.append(cid)
                if len(selected_ids) >= k_int:
                    break

            methods = []
            for cid in selected_ids[:k_int]:
                methods.extend(
                    df.loc[df["id"].astype(str) == cid, "method_name"].astype(str).tolist()
                )
            return methods

        for line in f:
            example_counter += 1
            data = json.loads(line.strip())
            deps = []
            target_method = data.get("namespace") or ""
            target_method_norm = _normalize_symbol(target_method)
            target_feature_id = method_norm_to_feature_id.get(target_method_norm)
            if target_feature_id is None:
                target_method_with_init = _maybe_insert_init_in_target_method(target_method_norm)
                if target_method_with_init != target_method_norm:
                    target_feature_id = method_norm_to_feature_id.get(target_method_with_init)
            target_feature_other_methods = []
            if target_feature_id is not None:
                target_feature_methods = (
                    df.loc[df["id"].astype(str) == str(target_feature_id), "method_name"]
                    .astype(str)
                    .tolist()
                )
                target_feature_other_methods = [
                    _normalize_symbol(m) for m in target_feature_methods if _normalize_symbol(m) != target_method_norm
                ]

            target_method_sig = resolve_method_signature(target_method)
            target_method_code = method_sig_to_code.get(target_method_sig, "")
            if not target_method_code:
                target_method_with_init = _maybe_insert_init_in_target_method(target_method_norm)
                if target_method_with_init != target_method_norm:
                    target_method_sig_with_init = resolve_method_signature(target_method_with_init)
                    target_method_code = method_sig_to_code.get(target_method_sig_with_init, "")
                    if target_method_code:
                        target_method_sig = target_method_sig_with_init
            similar_methods = {}
            target_code_tokens = tokenize_code(target_method_code) if target_method_code else []
            if target_code_tokens:
                target_code_scores = bm25_code.get_scores(target_code_tokens)
                target_code_order = np.argsort(target_code_scores)
                target_method_norm = _normalize_symbol(target_method)
                for k in TOP_SM:
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
                for k in TOP_SM:
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
            #input("please confirm the query!")
            query_embedding = model.encode([query])
            similarities = cosine_similarity(query_embedding, cluster_embeddings)[0]

            for k in CLUSTER_KS:
                methods_k = methods_from_top_k_clusters(similarities, k, target_feature_id)
                methods_k = filter_out_target_method(methods_k, target_method)
                cluster_metrics[k]["pred"] += len(methods_k)
                cluster_metrics[k]["match"] += compute_task_recall(deps, build_context_code_list(methods_k))["dependency_hit"]

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
            for k in CLUSTER_KS:
                mk = methods_from_top_k_clusters(similarities, k, target_feature_id)
                mk = filter_out_target_method(mk, target_method)
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

        out_dir = os.path.dirname(FILTERED_PATH)
        feature_path = os.path.join(out_dir, "diagnostic_***feature.jsonl")
        with open(feature_path, "w", encoding="utf-8") as fo:
            for rec in feature_records:
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
        
        # Save the updated cache
        with open(refined_queries_cache_path, 'w') as f:
            json.dump(refined_queries_cache, f, indent=4)

        # ===================== DIAGNOSTIC JSON CODE END =====================
        def safe_div(a, b):
            return (a / b) if b != 0 else 0

        for k in CLUSTER_KS:
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
    
    print(f"Analysis completed for {project_dir}")


if __name__ == "__main__":
    analyze_project(PROJECT_DIR)
