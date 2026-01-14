import os
import json
import pandas as pd
import networkx as nx
from collections import defaultdict


METHODS_CSV = "/data/data_public/riverbag/testRepoSummaryOut/mrjob/1:3/methods.csv"
ENRE_JSON = "/data/data_public/riverbag/testRepoSummaryOut/mrjob/1:3/mrjob-report-enre.json"
FILTERED_PATH = "/data/data_public/riverbag/testRepoSummaryOut/mrjob/1:3/filtered.jsonl"
OUTPUT_GRAPH_PATH = "/data/data_public/riverbag/testRepoSummaryOut/mrjob/1:3/bm25code_graph_results" 
ANALYSIS_OUTPUT_FILE = "/data/data_public/riverbag/testRepoSummaryOut/mrjob/1:3/bm25code_analysis_diff_report.txt"

REMOVE_FIRST_DOT_PREFIX = True
PREFIX = "mrjob"  # 如果移除前缀的选项为True，这里记得指定项目的名称作为前缀

def load_methods_csv(csv_path):
    df = pd.read_csv(csv_path)
    # method_signature -> info
    mapping = {}
    for _, row in df.iterrows():
        mapping[row['method_signature']] = row.to_dict()
    return mapping

def load_enre_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    variables = data.get('variables', [])
    cells = data.get('cells', [])
    
    valid_nodes = {} # id -> data
    qname_to_id = {} # id -> qualifiedName
    id_to_qname = {} # qualifiedName -> id
    
    for var in variables:
        if not var.get('category', '').startswith('Un'):
            vid = var['id']
            qname = var['qualifiedName']
            valid_nodes[vid] = var
            qname_to_id[qname] = vid
            id_to_qname[vid] = qname
            
    # Build adjacency
    adj = defaultdict(list) # src -> [(dest, kind)]
    reverse_adj = defaultdict(list) # dest -> [(src, kind)]
    seen_edges = set()

    for cell in cells:
        src = cell.get('src')
        dest = cell.get('dest')
        kind = cell.get('values', {}).get('kind')
        if src in valid_nodes and dest in valid_nodes:
            edge_tuple = (src, dest, kind)
            if edge_tuple not in seen_edges:
                seen_edges.add(edge_tuple)
                adj[src].append((dest, kind))
                reverse_adj[dest].append((src, kind))
            
    return valid_nodes, qname_to_id, id_to_qname, adj, reverse_adj

def normalize_signature(sig, remove_prefix=False):    
    # clean = sig.split('(')[0]
    if remove_prefix and '.' in sig:
        # Remove first segment
        parts = sig.split('.')
        if len(parts) > 1:
            clean = '.'.join(parts[1:])
    return clean

def main():
    # Ensure output directory exists
    output_dir = os.path.dirname(ANALYSIS_OUTPUT_FILE)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(ANALYSIS_OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        def log(msg):
            print(msg)
            f_out.write(msg + "\n")
            
        log("Loading Data...")
        method_map = load_methods_csv(METHODS_CSV)
        enre_nodes, qname_to_id, id_to_qname, adj, reverse_adj = load_enre_json(ENRE_JSON)
        
        tasks = []
        with open(FILTERED_PATH, 'r') as f:
            for line in f:
                if line.strip():
                    tasks.append(json.loads(line))
                    
        log(f"Loaded {len(tasks)} tasks.")
        
        # Iterate tasks
        for i, task in enumerate(tasks):
            task_id = task.get('example_id', i + 1)
            
            # Ground Truth
            dependency = task.get('dependency', {})
            gt_methods = []
            gt_methods.extend(dependency.get('intra_class', []))
            gt_methods.extend(dependency.get('intra_file', []))
            gt_methods.extend(dependency.get('cross_file', []))
            
            # Load Graph
            gml_filename = f"task_{task_id}_ori.gml"
            gml_path = os.path.join(OUTPUT_GRAPH_PATH, gml_filename)
            if not os.path.exists(gml_path):            
                log(f"Graph file not found for task {task_id}: {gml_path}")
                continue
            
            try:
                G = nx.read_gml(gml_path)
            except Exception as e:
                log(f"Error reading GML for task {task_id}: {e}")
                continue
                
            # Get graph nodes' signatures
            graph_sigs = set()
            node_id_map = {} # normalized_sig -> graph_node_id (ENRE ID)
            
            for node_id, data in G.nodes(data=True):
                sig = data.get('sig')
                # Normalize
                norm_sig = normalize_signature(sig, remove_prefix=REMOVE_FIRST_DOT_PREFIX)
                graph_sigs.add(norm_sig)
                
                # We assume the node ID in NetworkX IS the ENRE ID (as I implemented in build_graph)
                # G.nodes() yields the ID.
                node_id_map[norm_sig] = node_id
                
            # Compare GT with Graph
            missing_methods = []
            for gt in gt_methods:            
                if gt not in graph_sigs:
                    missing_methods.append(gt)
                    
            log(f"Task {task_id}: {len(missing_methods)} missing methods out of {len(gt_methods)} GT.")
            
            if not missing_methods:
                continue
                
            # Analyze missing methods
            for missing in missing_methods:
                # 1. Map to ENRE element
                # Try to find ENRE ID for 'missing'
                # Missing is like "mrjob.hadoop..."
                # ENRE qnames are like "mrjob.mrjob.hadoop..." (sometimes doubled prefix)
                # We need to match `missing` to keys in `qname_to_id`.
                
                found_eid = None
    
                if REMOVE_FIRST_DOT_PREFIX:
                    missing = PREFIX + '.' + missing
                
                # Exact match check
                if missing in qname_to_id:
                    found_eid = qname_to_id[missing]
                else:
                    # Fuzzy match: check if any qname ends with missing or vice versa
                    # ENRE: mrjob.mrjob.x.y
                    # Missing: mrjob.x.y
                    # Check if qname.endswith("." + missing) or qname == "mrjob." + missing
                    # candidates = []
                    # for qname, eid in qname_to_id.items():
                    #     # Check suffix match
                    #     if qname == missing or qname.endswith("." + missing):
                    #         candidates.append(eid)
                    #     # Check if missing is suffix of qname (common if package prefix missing in GT)
                    #     elif missing in qname and qname.split('.')[-len(missing.split('.')):] == missing.split('.'):
                    #          candidates.append(eid)
                    
                    # if candidates:
                    #     # Pick best? First?
                    #     found_eid = candidates[0]
                    pass
                    
                
                if found_eid is None:
                    log(f"  [Missing] {missing} -> Could not find in ENRE.")
                    continue
                    
                log(f"  [Missing] {missing} -> ENRE ID {found_eid} ({id_to_qname[found_eid]}) ({enre_nodes[found_eid].get('category')})")
                
                # 2. Find relations with EXISTING graph nodes
                # Check relations in ENRE (adj)
                # Outgoing from missing
                if found_eid in adj:
                    for dest, kind in adj[found_eid]:
                        dest = str(dest)
                        if dest in G.nodes:
                            dest_sig = G.nodes[dest].get('sig', G.nodes[dest].get('label'))
                            log(f"    -> Relation: [Missing] --({kind})--> [Graph Node {dest}] {G.nodes[dest].get('category')} ({dest_sig})")
                            
                # Incoming to missing
                if found_eid in reverse_adj:
                    for src, kind in reverse_adj[found_eid]:
                        src = str(src)
                        if src in G.nodes:
                            src_sig = G.nodes[src].get('sig', G.nodes[src].get('label'))
                            log(f"    -> Relation: [Graph Node {src}] {G.nodes[src].get('category')} ({src_sig}) --({kind})--> [Missing]")

if __name__ == "__main__":
    main()
