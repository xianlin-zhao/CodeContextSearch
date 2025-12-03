import pandas as pd
import json
import networkx as nx
# import matplotlib.pyplot as plt # Not essential for just building/saving, but good if visualization needed
import os
from collections import defaultdict


METHODS_CSV = "/data/zxl/Search2026/outputData/repoSummaryOut/mrjob/1112_codet5/methods.csv"
ENRE_JSON = "/data/zxl/Search2026/CodeContextSearch/src/summarization/mrjob-report-enre.json"
FILTERED_PATH = "/data/zxl/Search2026/outputData/repoSummaryOut/mrjob/1112_codet5/filtered.jsonl" 
DIAGNOSTIC_JSONL = "/data/zxl/Search2026/outputData/repoSummaryOut/mrjob/1112_codet5/diagnostic_feature.jsonl"
OUTPUT_GRAPH_PATH = "/data/zxl/Search2026/outputData/devEvalSearchOut/Internet_boto/graph_results"


# 读入之前处理的所有method信息
print("Loading METHODS_CSV...")
df_methods = pd.read_csv(METHODS_CSV)
method_sig_to_info = {}
for index, row in df_methods.iterrows():
    # Ensure we handle potential NaN values or convert to string if needed
    sig = str(row['method_signature'])
    method_sig_to_info[sig] = row.to_dict()

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
    
    if not os.path.exists(OUTPUT_GRAPH_PATH):
        os.makedirs(OUTPUT_GRAPH_PATH)
        
    for i, rec in enumerate(diag_records):
        example_id = rec.get('example_id', i)
        
        # 获取搜索出来的所有method
        preds = []
        try:
            preds = rec["feature"]["top3"]["predictions"]
        except KeyError:
            print(f"Skipping task {example_id}: structure not matching data['feature']['top3']['predictions']")
            continue
            
        # 只要method签名
        pred_signatures = [p['method'] for p in preds]
        
        # Create a graph for this task
        G = nx.DiGraph()
        
        # Find corresponding ENRE IDs
        relevant_ids = set()
        for sig in pred_signatures:
            # strip parameters from sig to match ENRE.
            if '(' in sig:
                clean_sig = sig.split('(')[0]
            else:
                clean_sig = sig
                
            if clean_sig in qname_to_id:
                eid = qname_to_id[clean_sig]
                relevant_ids.add(eid)
                node_info = valid_nodes[eid]
                
                attrs = {
                    'sig': clean_sig,
                    'category': node_info['category']
                }
                
                # Find matching method info
                if sig in method_sig_to_info:
                    csv_info = method_sig_to_info[sig]
                    attrs['method_signature'] = str(csv_info.get('method_signature', ''))
                    attrs['func_file'] = str(csv_info.get('func_file', ''))
                    attrs['method_code'] = str(csv_info.get('method_code', ''))
                else:
                    print(f"Warning: No method info found for {sig}")
                    # Fallback: try to find by clean name if exact sig match fails
                    # This might be ambiguous if overloaded, but Python...
                    # found = False
                    # for k, v in method_sig_to_info.items():
                    #     if k.startswith(clean_sig + '('):
                    #         attrs['method_signature'] = str(v.get('method_signature', ''))
                    #         attrs['func_file'] = str(v.get('func_file', ''))
                    #         attrs['method_code'] = str(v.get('method_code', ''))
                    #         found = True
                    #         break
                    # if not found:
                    #     # print(f"Warning: No CSV info found for {sig}")
                    #     pass

                # Add node with extended attributes
                G.add_node(eid, **attrs)

        # Now add edges between these relevant nodes       
        for u in relevant_ids:
            if u in adj:
                for v, kind in adj[u]:
                    if v in relevant_ids:
                        G.add_edge(u, v, type=kind)
        
        # Save graph
        out_file = os.path.join(OUTPUT_GRAPH_PATH, f"task_{example_id}_ori.gml")
        nx.write_gml(G, out_file)
        print(f"Saved graph for task {example_id} with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")


if __name__ == "__main__":
    build_graph()
