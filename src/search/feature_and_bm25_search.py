import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from rank_bm25 import BM25Okapi

FEATURE_CSV = "/data/zxl/Search2026/outputData/repoSummaryOut/mrjob/1111/mrjob_functionName_features.csv"
METHODS_CSV = "/data/zxl/Search2026/outputData/repoSummaryOut/mrjob/1111/methods.csv"
FILTERED_PATH = "/data/zxl/Search2026/outputData/repoSummaryOut/mrjob/1111/filtered.jsonl"
# DevEval数据集case的路径（json，不是数据集项目本身）
DATA_JSONL = "/data/lowcode_public/DevEval/data_have_dependency_cross_file.jsonl"


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
    df = pd.read_csv(FEATURE_CSV)
    clusters = df.groupby('id')['desc'].first().reset_index()
    # 提取clusters中所有的method_name
    method_names = df['method_name'].str.split('(').str[0].unique().tolist()

    #model = SentenceTransformer('all-mpnet-base-v2')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    cluster_embeddings = model.encode(clusters['desc'].tolist())

    # load methods corpus
    methods_df = pd.read_csv(METHODS_CSV, dtype=str).fillna('')
    # ensure columns exist
    if 'method_signature' not in methods_df.columns or 'method_code' not in methods_df.columns:
        raise ValueError("methods.csv must contain 'method_signature' and 'method_code' columns")

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

    signature_docs = [tokenize_signature(s) for s in methods_df['method_signature'].tolist()]
    print(signature_docs[:10])
    print("=======================================")
    code_docs = [tokenize_code(c) for c in methods_df['method_code'].tolist()]
    print(code_docs[:10])
    # store the original method strings to return as predicted items
    methods_corpus_strings = methods_df['method_signature'].tolist()

    bm25_sig = BM25Okapi(signature_docs)
    bm25_code = BM25Okapi(code_docs)

    with open(FILTERED_PATH, 'r') as f:
        top_gt = 0

        cluster_ks = [1, 3, 5]
        cluster_metrics = {k: {"match": 0, "pred": 0} for k in cluster_ks}

        sig_ks = [5, 10, 15]
        sig_metrics = {k: {"match": 0, "pred": 0} for k in sig_ks}

        code_metrics = {k: {"match": 0, "pred": 0} for k in sig_ks}

        def methods_from_top_k_clusters(similarities, k):
            idx = np.argsort(similarities)[-k:][::-1]
            ids = clusters.iloc[idx]["id"].tolist()
            methods = []
            for cid in ids:
                methods.extend(df[df["id"] == cid]["method_name"].tolist())
            return methods

        def bm25_topk_strings(order, k):
            idx = order[-k:][::-1]
            return [methods_corpus_strings[i] for i in idx]

        for line in f:
            data = json.loads(line.strip())
            deps = []
            deps.extend(data['dependency']['intra_class'])
            deps.extend(data['dependency']['intra_file'])
            deps.extend(data['dependency']['cross_file'])
            deps = [dep for dep in deps if dep in method_names]

            top_gt += len(deps)

            # feature-based search
            query = data['requirement']['Functionality'] + ' ' + data['requirement']['Arguments']
            query_embedding = model.encode([query])
            similarities = cosine_similarity(query_embedding, cluster_embeddings)[0]

            for k in cluster_ks:
                methods_k = methods_from_top_k_clusters(similarities, k)
                cluster_metrics[k]["pred"] += len(methods_k)
                cluster_metrics[k]["match"] += sum(1 for dep in deps if any(dep in m for m in methods_k))

            # signature-based search using BM25
            q_tokens = re.findall(r'\w+', query.lower())
            sig_scores = bm25_sig.get_scores(q_tokens)
            sig_order = np.argsort(sig_scores)
            for k in sig_ks:
                m = bm25_topk_strings(sig_order, k)
                sig_metrics[k]["pred"] += len(m)
                sig_metrics[k]["match"] += sum(1 for dep in deps if any(dep in s for s in m))

            # code-based search using BM25
            code_scores = bm25_code.get_scores(q_tokens)
            code_order = np.argsort(code_scores)
            for k in sig_ks:
                m = bm25_topk_strings(code_order, k)
                code_metrics[k]["pred"] += len(m)
                code_metrics[k]["match"] += sum(1 for dep in deps if any(dep in s for s in m))

        def safe_div(a, b):
            return (a / b) if b != 0 else 0

        for k in cluster_ks:
            m = cluster_metrics[k]["match"]
            p = cluster_metrics[k]["pred"]
            print(f"Top {k} Match: {m}")
            print(f"Top {k} Pred: {p}")
            print(f"Top {k} P={(safe_div(m, p))*100:.2f}%")
            print(f"Top {k} R={(safe_div(m, top_gt))*100:.2f}%")
            print(f"Top {k} F1={(2* safe_div(m, p) * safe_div(m, top_gt) / (safe_div(m, p) + safe_div(m, top_gt)))*100:.2f}%")
            print("--------------------------------")

        # ---- print BM25 signature-based metrics ----
        print("BM25 (signature) results:")
        for k in sig_ks:
            m = sig_metrics[k]["match"]
            p = sig_metrics[k]["pred"]
            print(f"Top{k} Match: {m}, Pred: {p}, P={(safe_div(m, p))*100:.2f}%, R={(safe_div(m, top_gt))*100:.2f}%")
        print("--------------------------------")

        # ---- print BM25 code-based metrics ----
        print("BM25 (code) results:")
        for k in sig_ks:
            m = code_metrics[k]["match"]
            p = code_metrics[k]["pred"]
            print(f"Top{k} Match: {m}, Pred: {p}, P={(safe_div(m, p))*100:.2f}%, R={(safe_div(m, top_gt))*100:.2f}%")
        print("--------------------------------")
    
    print(f"Analysis completed for {project_path}")


if __name__ == "__main__":
    project_path = "System/mrjob"
    analyze_project(project_path)
