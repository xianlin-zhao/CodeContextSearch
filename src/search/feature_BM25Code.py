import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
from rank_bm25 import BM25Okapi
from tqdm import tqdm

# PROJECT_PATH = "Internet/boto"
PROJECT_PATH = "System/mrjob"


FEATURE_CSV = "/data/zxl/Search2026/outputData/repoSummaryOut/mrjob/1112_codet5/features.csv" 
METHODS_CSV = "/data/zxl/Search2026/outputData/repoSummaryOut/mrjob/1112_codet5/methods.csv" 
FILTERED_PATH = "/data/zxl/Search2026/outputData/repoSummaryOut/mrjob/1112_codet5/filtered.jsonl" 

# FEATURE_CSV = "/home/riverbag/testRepoSummaryOut/mrjob/1122_codet5/features.csv" 
# METHODS_CSV = "/home/riverbag/testRepoSummaryOut/mrjob/1122_codet5/methods.csv" 
# FILTERED_PATH = "/home/riverbag/testRepoSummaryOut/mrjob/1122_codet5/filtered.jsonl" 

# FEATURE_CSV = "/home/riverbag/testRepoSummaryOut/boto/boto_testAug/1122_codet5/features.csv" 
# METHODS_CSV = "/home/riverbag/testRepoSummaryOut/boto/boto_testAug/1122_codet5/methods.csv" 
# FILTERED_PATH = "/home/riverbag/testRepoSummaryOut/boto/boto_testAug/1122_codet5/filtered.jsonl" 
# DevEval数据集case的路径（json，不是数据集项目本身）
DATA_JSONL = "/data/lowcode_public/DevEval/data_have_dependency_cross_file.jsonl"

# 是否需要把method名称规范化，例如得到的csv中是mrjob.mrjob.xx，将其规范化为mrjob.xx，以便进行测评
NEED_METHOD_NAME_NORM = True

# Recall parameters
# RECALL_CLUSTER_K = 5  # Use top 5 clusters for recall
CLUSTER_KS = [1, 3, 5]
SIG_KS = [5, 10, 15]

def analyze_project(project_path):
    # Step 1: Filter by project_path
    print("Filtering data...")
    with open(DATA_JSONL, 'r') as infile, open(FILTERED_PATH, 'w') as outfile:
        for line in infile:
            data = json.loads(line.strip())
            if data.get('project_path') == project_path:
                outfile.write(line)
    
    # Step 2: Load Features and Clusters
    print("Loading features...")
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
    print(f"Loaded {len(method_names)} unique method names from features.")
    print(method_names[:10])

    print("Encoding clusters...")
    # model = SentenceTransformer('all-mpnet-base-v2')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    cluster_embeddings = model.encode(clusters['desc'].tolist())

    # Step 3: Load Methods for BM25
    print("Loading methods for BM25...")
    methods_df = pd.read_csv(METHODS_CSV, dtype=str).fillna('')
    if 'method_signature' not in methods_df.columns or 'method_code' not in methods_df.columns:
        raise ValueError("methods.csv must contain 'method_signature' and 'method_code' columns")

    def tokenize_signature(sig: str):
        part = sig.split('.')[-1]
        part = part.replace('_', ' ')
        part = re.sub(r'[\(\),.:]', ' ', part)
        toks = re.findall(r'\w+', part.lower())
        return toks

    def tokenize_code(code: str):
        toks = re.findall(r'\w+', code.lower())
        return toks

    # Build BM25 on Code
    print("Building BM25 index on code...")
    code_docs = [tokenize_code(c) for c in methods_df['method_code'].tolist()]
    methods_corpus_strings = methods_df['method_signature'].tolist()
    bm25_code = BM25Okapi(code_docs)

    # Build Mapping: Feature Method Name -> Indices in methods_df
    # This is needed to map the "Recalled" methods from Feature Search to the BM25 corpus
    print("Building mapping from Feature names to Method indices...")
    feature_name_to_indices = {}
    
    # Pre-compute valid indices for each feature method name
    # Note: This might be slow if done naively. We assume method_name is a substring of method_signature.
    
    # Optimization: iterate over methods_df once
    # For each method in methods_df, check which feature names it matches.
    # To avoid O(N*M), we can try to match by exact string if possible, or use the containment assumption.
    # Given the previous code used `any(dep in m for m in methods_k)`, where dep is from feature names.
    # We need to support the same logic.
    
    # Let's try to do it:
    # 1. Get all unique feature method names (un-normalized for matching with raw data if needed, or normalized?)
    # The 'deps' in ground truth are filtered by `method_names` (which are normalized if NEED_METHOD_NAME_NORM is True).
    # But `methods_from_top_k_clusters` returns `df['method_name']` which is NOT normalized in the dataframe itself (unless we updated it).
    # In original code: `df['method_name_norm']` was created but `methods_from_top_k_clusters` uses `df["method_name"]`.
    # So the recall returns un-normalized names.
    
    raw_feature_names = df['method_name'].unique().tolist()
    
    # We will map RAW feature names to method indices.
    # Because `methods_from_top_k_clusters` returns raw names.
    
    # To speed up: 
    # many raw_feature_names might be substrings of methods_corpus_strings.
    # Let's just iterate. For 10k methods and 2k features, 20M checks is fine in Python (~seconds).
    
    for idx, sig in enumerate(methods_corpus_strings):
        for fname in raw_feature_names:
            if fname in sig:
                if fname not in feature_name_to_indices:
                    feature_name_to_indices[fname] = []
                feature_name_to_indices[fname].append(idx)
                
    print(f"Mapping built for {len(feature_name_to_indices)} feature names.")

    # Helper functions
    def methods_from_top_k_clusters(similarities, k):
        idx = np.argsort(similarities)[-k:][::-1]
        ids = clusters.iloc[idx]["id"].tolist()
        methods = []
        for cid in ids:
            methods.extend(df[df["id"] == cid]["method_name"].tolist())
        return list(set(methods)) # Dedup
       

    def safe_div(a, b):
        return (a / b) if b != 0 else 0

    # Evaluation
    print("Starting Hybrid Search Evaluation...")
    with open(FILTERED_PATH, 'r') as f:
        lines = f.readlines()

    hybrid_records = []
    
    # Metrics containers
    # metrics structure: metrics[cluster_k][sig_k] = {"match": 0, "pred": 0}
    metrics = {}
    for ck in CLUSTER_KS:
        metrics[ck] = {}
        for sk in SIG_KS:
            metrics[ck][sk] = {"match": 0, "pred": 0}
            
    top_gt = 0
    example_counter = 0

    for line in tqdm(lines):
        example_counter += 1
        data = json.loads(line.strip())
        deps = []
        deps.extend(data['dependency']['intra_class'])
        deps.extend(data['dependency']['intra_file'])
        deps.extend(data['dependency']['cross_file'])
        
        # Filter deps by the normalized method names (as in original code)
        deps = [dep for dep in deps if dep in method_names]
        top_gt += len(deps)

        # Feature Search Embedding
        query = data['requirement']['Functionality'] + ' ' + data['requirement']['Arguments']
        query_embedding = model.encode([query])
        similarities = cosine_similarity(query_embedding, cluster_embeddings)[0]
        
        q_tokens = re.findall(r'\w+', query.lower())
        all_bm25_scores = bm25_code.get_scores(q_tokens)
        
        # Prepare record for this example
        record = {
            "example_id": example_counter,
            "query": query,
            "ground_truth": deps,
            "hybrid": {}
        }

        # Iterate over different Cluster Recall Ks
        for ck in CLUSTER_KS:
            # 1. Feature Search Recall (Top ck clusters)
            recalled_raw_names = methods_from_top_k_clusters(similarities, ck)
            
            # Map to BM25 indices
            candidate_indices = set()
            for rname in recalled_raw_names:
                if rname in feature_name_to_indices:
                    candidate_indices.update(feature_name_to_indices[rname])
            
            candidate_indices = list(candidate_indices)
            
            # 2. BM25 Re-ranking on Code
            final_methods = []
            if candidate_indices:
                scored_candidates = []
                for idx in candidate_indices:
                    scored_candidates.append((all_bm25_scores[idx], methods_corpus_strings[idx]))
                
                # Sort by score desc
                scored_candidates.sort(key=lambda x: x[0], reverse=True)
                
                # Extract method signatures
                final_methods = [x[1] for x in scored_candidates]
            
            # 3. Calculate metrics for different Signature Top-Ks (SIG_KS)
            record["hybrid"][f"recall_top{ck}_clusters"] = {}
            
            for sk in SIG_KS:
                mk = final_methods[:sk]
                
                num_match = sum(1 for dep in deps if any(dep in m for m in mk))
                num_pred = len(mk)
                
                metrics[ck][sk]["pred"] += num_pred
                metrics[ck][sk]["match"] += num_match
                
                # Per-example metrics
                num_gt_ex = len(deps)
                p = safe_div(num_match, num_pred)
                r = safe_div(num_match, num_gt_ex)
                f1 = safe_div(2 * p * r, p + r)
                
                record["hybrid"][f"recall_top{ck}_clusters"][f"rank_top{sk}"] = {
                    "metrics": {"P": p, "R": r, "F1": f1, "pred": num_pred, "match": num_match, "gt": num_gt_ex},
                    "predictions": [{"method": m, "match": any(dep in m for dep in deps)} for m in mk]
                }
        
        hybrid_records.append(record)

    # Output results
    out_dir = os.path.dirname(FILTERED_PATH)
    hybrid_path = os.path.join(out_dir, "diagnostic_hybrid.jsonl")
    with open(hybrid_path, "w", encoding="utf-8") as fo:
        for rec in hybrid_records:
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("\nHybrid Search Results (Feature Recall + BM25 Code Re-rank):")
    print("Format: Recall Top-K Clusters -> Rank Top-K Results")
    print("="*60)
    
    for ck in CLUSTER_KS:
        print(f"--- Feature Search Recall: Top {ck} Clusters ---")
        for sk in SIG_KS:
            m = metrics[ck][sk]["match"]
            p = metrics[ck][sk]["pred"]
            
            precision = (safe_div(m, p)) * 100
            recall = (safe_div(m, top_gt)) * 100
            f1 = (2 * safe_div(m, p) * safe_div(m, top_gt) / (safe_div(m, p) + safe_div(m, top_gt))) * 100 if (safe_div(m, p) + safe_div(m, top_gt)) > 0 else 0
            
            print(f"  Rank Top {sk}:")
            print(f"    Match: {m}, Pred: {p}")
            print(f"    P={precision:.2f}%, R={recall:.2f}%, F1={f1:.2f}%")
        print("-" * 60)

    print(f"Hybrid analysis completed. Results saved to {hybrid_path}")

if __name__ == "__main__":
    analyze_project(PROJECT_PATH)
