import os
import json
import pandas as pd
import networkx as nx
from collections import defaultdict


METHODS_CSV = "/data/zxl/Search2026/outputData/repoSummaryOut/mrjob/1112_codet5/methods.csv"
ENRE_JSON = "/data/zxl/Search2026/CodeContextSearch/src/summarization/mrjob-report-enre.json"
FILTERED_PATH = "/data/zxl/Search2026/outputData/repoSummaryOut/mrjob/1112_codet5/filtered.jsonl"
OUTPUT_GRAPH_PATH = "/data/zxl/Search2026/outputData/devEvalSearchOut/System_mrjob/graph_results" 

REMOVE_FIRST_DOT_PREFIX = True

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
    
    for cell in cells:
        src = cell.get('src')
        dest = cell.get('dest')
        kind = cell.get('values', {}).get('kind')
        if src in valid_nodes and dest in valid_nodes:
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
    print("Loading Data...")
    method_map = load_methods_csv(METHODS_CSV)
    enre_nodes, qname_to_id, id_to_qname, adj, reverse_adj = load_enre_json(ENRE_JSON)
    
    tasks = []
    with open(FILTERED_PATH, 'r') as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
                
    print(f"Loaded {len(tasks)} tasks.")
    
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
            print(f"Graph file not found for task {task_id}: {gml_path}")
            continue
        
        try:
            G = nx.read_gml(gml_path)
        except Exception as e:
            print(f"Error reading GML for task {task_id}: {e}")
            continue
            
        # Get graph nodes' signatures
        graph_sigs = set()
        node_id_map = {} # normalized_sig -> graph_node_id (ENRE ID)
        
        for node_id, data in G.nodes(data=True):
            # label is ENRE ID? User said "label是指它在ENRE_REPORT里面的编号"
            # Wait, usually label is string name, ID is the node ID.
            # In build_graph.py: G.add_node(eid, label=clean_sig, ...)
            # So label was the signature string.
            # But user prompt says: "label refers to its ID in ENRE_REPORT". 
            # This might mean the `id` attribute or the `label` attribute.
            # In GML, `id` is implicit/explicit. 
            # If user says "label is ID", maybe they mean the visible label on the node IS the ID?
            # Or maybe they mean the `label` attribute holds the ID?
            # BUT later they say "especially label and sig attributes".
            # If label is ID, then what is sig?
            # In my build_graph, I saved `label` as clean_sig.
            # Let's look at the user's GML snippet:
            # `method_signature "mrjob.mrjob.runner..."`
            # `label "1"` (in my test output) or `label "mrjob..."`?
            # In the test_gml output: `label "1"`.
            # In my build_graph: `G.add_node(eid, label=clean_sig)` -> label would be string.
            # If user says "label is ID", maybe they are using a different graph generation script?
            # "Analysis... referring to build_graph" -> maybe build_graph *should have* put ID as label?
            # Whatever, I will trust the `sig` or `method_signature` attribute for the name.
            
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
            # Normalize GT? GT is usually "mrjob.xxx"
            # If remove prefix is on, we should remove it from GT too if it matches pattern?
            # Or maybe GT is already "clean"?
            # "mrjob.hadoop.HadoopJobRunner._hadoop_log_dirs"
            # If prefix "mrjob." is removed -> "hadoop..."
            # Let's try both raw and normalized match
            # norm_gt = normalize_signature(gt, remove_prefix=REMOVE_FIRST_DOT_PREFIX)
            
            if gt not in graph_sigs:
                missing_methods.append(gt)
                
        print(f"Task {task_id}: {len(missing_methods)} missing methods out of {len(gt_methods)} GT.")
        
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

            # 临时：在前面加上mrjob.
            missing = 'mrjob.' + missing
            
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
                print(f"  [Missing] {missing} -> Could not find in ENRE.")
                continue
                
            print(f"  [Missing] {missing} -> ENRE ID {found_eid} ({id_to_qname[found_eid]})")
            
            # 2. Find relations with EXISTING graph nodes
            # Check relations in ENRE (adj)
            # Outgoing from missing
            if found_eid in adj:
                for dest, kind in adj[found_eid]:
                    dest = str(dest)
                    if dest in G.nodes:
                        dest_sig = G.nodes[dest].get('sig', G.nodes[dest].get('label'))
                        print(f"    -> Relation: [Missing] --({kind})--> [Graph Node {dest}] ({dest_sig})")
                        
            # Incoming to missing
            if found_eid in reverse_adj:
                for src, kind in reverse_adj[found_eid]:
                    src = str(src)
                    if src in G.nodes:
                        src_sig = G.nodes[src].get('sig', G.nodes[src].get('label'))
                        print(f"    -> Relation: [Graph Node {src}] ({src_sig}) --({kind})--> [Missing]")

if __name__ == "__main__":
    main()
