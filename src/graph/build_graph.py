import pandas as pd
import json
import networkx as nx
# import matplotlib.pyplot as plt # Not essential for just building/saving, but good if visualization needed
import os
from collections import defaultdict


METHODS_CSV = "/data/data_public/riverbag/testRepoSummaryOut/211/diffprivlib/methods.csv"
ENRE_JSON = "/data/data_public/riverbag/testRepoSummaryOut/211/diffprivlib/diffprivlib-report-enre.json"
FILTERED_PATH = "/data/data_public/riverbag/testRepoSummaryOut/211/diffprivlib/filtered.jsonl" 
DIAGNOSTIC_JSONL = "/data/data_public/riverbag/testRepoSummaryOut/211/diffprivlib/diagnostic_feature.jsonl"
# OUTPUT_GRAPH_PATH = "/data/zxl/Search2026/outputData/devEvalSearchOut/Internet_boto/0115/graph_results"
OUTPUT_GRAPH_PATH = "/data/data_public/riverbag/testRepoSummaryOut/211/diffprivlib/graph_results"

REMOVE_FIRST_DOT_PREFIX = False
PREFIX = "diffprivlib"  # 如果移除前缀的选项为True，这里记得指定项目的名称作为前缀
#用来控制我们选几个相似的method（用来后面的类调用链预测加分的）
SIMILAR_TOPK = 3


# 读入之前处理的所有method信息
print("Loading METHODS_CSV...")
df_methods = pd.read_csv(METHODS_CSV)
method_sig_to_info = {}
method_clean_sig_to_info = {}
for index, row in df_methods.iterrows():
    # Ensure we handle potential NaN values or convert to string if needed
    sig = str(row['method_signature'])
    method_sig_to_info[sig] = row.to_dict()
    clean_sig = sig.split("(", 1)[0] if "(" in sig else sig
    method_clean_sig_to_info[clean_sig] = row.to_dict()
print(f"Loaded {len(method_sig_to_info)} methods from CSV.")


# 读入之前生成的enre_report.json，处理代码元素及其相互关系
print("Loading ENRE_JSON and processing variables...")
with open(ENRE_JSON, 'r') as f:
    enre_data = json.load(f)

variables = enre_data.get('variables', [])
cells = enre_data.get('cells', [])

valid_nodes = {} # id -> node_data
id_to_qname = {} # id -> qualifiedName
qname_to_id = {} # qualifiedName -> id

for var in variables:
    cat = var.get('category', '')
    # 如果category是Unresolved或Unresolved开头，则忽略
    if not cat.startswith('Un'):
        vid = var['id']
        qname = var['qualifiedName']
        
        valid_nodes[vid] = var
        id_to_qname[vid] = qname
        qname_to_id[qname] = vid
        
print(f"Filtered {len(valid_nodes)} valid nodes from ENRE_JSON.")


print("Processing relations in ENRE_JSON...")
edges = []
seen_edges = set()
for cell in cells:
    src = cell.get('src')
    dest = cell.get('dest')
    values = cell.get('values', {})
    kind = values.get('kind')
    
    # 把所有有效边加入edges
    if src in valid_nodes and dest in valid_nodes:
        edge_tuple = (src, dest, kind)
        if edge_tuple not in seen_edges:
            seen_edges.add(edge_tuple)
            edges.append((src, dest, kind))
        
print(f"Loaded {len(edges)} valid edges.")


print("Graph construction logic needs adjacency list...")
# Build adjacency for whole graph
adj = defaultdict(list)
for src, dest, kind in edges:
    adj[src].append((dest, kind))

def _maybe_insert_init_in_target_method(target_method):
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

def build_graph():
    print("Step 1: Loading FILTERED_PATH for task info...")
    tasks_info = []
    # 读入一个项目的所有任务，每行是一个任务
    with open(FILTERED_PATH, 'r') as f:
        for line in f:
            if line.strip():
                tasks_info.append(json.loads(line))
    print(f"Loaded {len(tasks_info)} tasks from FILTERED_PATH.")

    print("Step 2 Loading DIAGNOSTIC_JSONL and Building Graph...")
    diag_records = []
    # 读入之前搜索的结果（包含是否match等指标）
    with open(DIAGNOSTIC_JSONL, 'r') as f:
        for line in f:
            if line.strip():
                diag_records.append(json.loads(line))
    
    os.makedirs(OUTPUT_GRAPH_PATH, exist_ok=True)

    total_tasks = len(diag_records)
    tasks_same_file_nonzero = 0
    tasks_same_feature_nonzero = 0
    tasks_similar_method_nonzero = 0
        
    for i, rec in enumerate(diag_records):
        example_id = rec.get('example_id', i)
        target_method = tasks_info[i].get("namespace") or ""
        if REMOVE_FIRST_DOT_PREFIX:
            target_method = PREFIX + "." + target_method
        target_feature_other_methods = rec.get("target_feature_other_methods") or []
        target_feature_other_methods_set = {
            (x.split("(", 1)[0] if "(" in str(x) else str(x))
            for x in target_feature_other_methods
            if x
        }

        similar_methods = rec.get("similar_methods")
        if not isinstance(similar_methods, dict):
            similar_methods = {}
        similar_method_names = similar_methods.get(f"top{SIMILAR_TOPK}") or []
        similar_method_names_set = {
            (x.split("(", 1)[0] if "(" in str(x) else str(x))
            for x in similar_method_names
            if x
        }

        target_file = ""
        if target_method in method_clean_sig_to_info:
            target_file = str(method_clean_sig_to_info[target_method].get("func_file", ""))
        if not target_file:
            target_method_with_init = _maybe_insert_init_in_target_method(target_method)
            if target_method_with_init != target_method and target_method_with_init in method_clean_sig_to_info:
                target_file = str(method_clean_sig_to_info[target_method_with_init].get("func_file", ""))
        print(f"[task] example_id={example_id} target_method={target_method} target_file={target_file}", flush=True)

        preds = []
        try:
            preds = rec["feature"]["top3"]["predictions"]
            filtered_preds = [
                p
                for p in preds
                if (p["method"] if "(" not in p["method"] else p["method"].split("(", 1)[0])
                != target_method
            ]
            if len(filtered_preds) != len(preds):
                print(
                    f"Attention! {len(preds) - len(filtered_preds)} items filtered out for task {example_id} (method == {target_method})",
                    flush=True,
                )
            preds = filtered_preds
        except KeyError:
            print(f"Skipping task {example_id}: missing feature/top3/predictions", flush=True)
            continue

        pred_signatures = [p["method"] for p in preds]

        G = nx.DiGraph()
        G.graph["target_method"] = target_method
        G.graph["target_file"] = target_file

        relevant_ids = set()
        for sig in pred_signatures:
            clean_sig = sig.split("(", 1)[0] if "(" in sig else sig

            if clean_sig in qname_to_id:
                eid = qname_to_id[clean_sig]
                relevant_ids.add(eid)
                node_info = valid_nodes[eid]

                attrs = {
                    'sig': clean_sig,
                    'category': node_info['category'],
                    'is_SameFile': False,
                    'is_SameFeature': False,
                    'is_SimilarMethod': False,
                }

                if sig in method_sig_to_info:
                    csv_info = method_sig_to_info[sig]
                    attrs['method_signature'] = str(csv_info.get('method_signature', ''))
                    attrs['func_file'] = str(csv_info.get('func_file', ''))
                    attrs['method_code'] = str(csv_info.get('method_code', ''))
                elif clean_sig in method_sig_to_info:
                    csv_info = method_sig_to_info[clean_sig]
                    attrs['method_signature'] = str(csv_info.get('method_signature', ''))
                    attrs['func_file'] = str(csv_info.get('func_file', ''))
                    attrs['method_code'] = str(csv_info.get('method_code', ''))
                else:
                    print(f"Warning: No method info found for {sig}", flush=True)

                G.add_node(eid, **attrs)

        for u in relevant_ids:
            if u in adj:
                for v, kind in adj[u]:
                    if v in relevant_ids:
                        G.add_edge(u, v, type=kind)

        same_file_count = 0
        same_feature_count = 0
        similar_method_count = 0
        if target_file:
            for _, attrs in G.nodes(data=True):
                if str(attrs.get("func_file", "")) == target_file:
                    attrs["is_SameFile"] = True
                    same_file_count += 1
        if target_feature_other_methods_set:
            for _, attrs in G.nodes(data=True):
                if str(attrs.get("sig", "")) in target_feature_other_methods_set:
                    attrs["is_SameFeature"] = True
                    same_feature_count += 1
        if similar_method_names_set:
            for _, attrs in G.nodes(data=True):
                if str(attrs.get("sig", "")) in similar_method_names_set:
                    attrs["is_SimilarMethod"] = True
                    similar_method_count += 1
        print(
            f"[task] example_id={example_id} same_file={same_file_count} same_feature={same_feature_count} similar_method={similar_method_count}",
            flush=True,
        )
        if same_file_count != 0:
            tasks_same_file_nonzero += 1
        if same_feature_count != 0:
            tasks_same_feature_nonzero += 1
        if similar_method_count != 0:
            tasks_similar_method_nonzero += 1

        out_file = os.path.join(OUTPUT_GRAPH_PATH, f"task_{example_id}_ori.gml")
        nx.write_gml(G, out_file)
        print(f"Saved graph for task {example_id} with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    print(
        f"[summary] total_tasks={total_tasks} same_file_nonzero_tasks={tasks_same_file_nonzero} same_feature_nonzero_tasks={tasks_same_feature_nonzero} similar_method_nonzero_tasks={tasks_similar_method_nonzero}",
        flush=True,
    )

if __name__ == "__main__":
    build_graph()
