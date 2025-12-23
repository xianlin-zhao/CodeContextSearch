import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import sys
sys.path.append("..")
import json
import pandas as pd
import networkx as nx
from collections import defaultdict
import torch
import torch.nn.functional as F
import numpy as np
from search.search_models.unixcoder import UniXcoder

METHODS_CSV = "/data/zxl/Search2026/outputData/repoSummaryOut/mrjob/1112_codet5/methods.csv"
ENRE_JSON = "/data/zxl/Search2026/CodeContextSearch/src/summarization/mrjob-report-enre.json"
FILTERED_PATH = "/data/zxl/Search2026/outputData/repoSummaryOut/mrjob/1112_codet5/filtered.jsonl"
OUTPUT_GRAPH_PATH = "/data/zxl/Search2026/outputData/devEvalSearchOut/System_mrjob/graph_results"
PROJECT_PATH = "/data/lowcode_public/DevEval/Source_Code/System"

def load_unixcoder_model(model_path_or_name=None, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = UniXcoder(model_path_or_name if model_path_or_name else "microsoft/unixcoder-base")
        model.to(device)
        model.eval()
        return model, device
    except Exception:
        raise ImportError("Unable to import UniXcoder. Ensure UniXcoder is installed / in PYTHONPATH.")

def encode_nl_with_unixcoder(model, device, text, max_length=512):
    tokens_ids = model.tokenize([text], max_length=max_length, mode="<encoder-only>")
    source_ids = torch.tensor(tokens_ids).to(device)
    with torch.no_grad():
        _, nl_embedding = model(source_ids)
        nl_embedding = F.normalize(nl_embedding, p=2, dim=1)
    return nl_embedding.cpu()

def encode_code_with_unixcoder(model, device, code_list, batch_size=32, max_length=512):
    all_embs = []
    for i in range(0, len(code_list), batch_size):
        batch = code_list[i:i+batch_size]
        # Tokenize individually to handle variable lengths
        batch_token_ids = []
        for code in batch:
             # Ensure code is string
            if not isinstance(code, str):
                code = ""
            ids = model.tokenize([code], max_length=max_length, mode="<encoder-only>")
            batch_token_ids.append(ids[0]) # ids is list of list
            
        # Pad to max length in batch
        max_len_in_batch = max(len(x) for x in batch_token_ids)
        padded_ids = []
        for x in batch_token_ids:
            padded_ids.append(x + [model.config.pad_token_id] * (max_len_in_batch - len(x)))
            
        source_ids = torch.tensor(padded_ids).to(device)
        
        with torch.no_grad():
            _, code_embedding = model(source_ids)
            normed = F.normalize(code_embedding, p=2, dim=1)
            all_embs.append(normed.cpu())
            
    if len(all_embs) == 0:
        return torch.empty((0,0))
    return torch.cat(all_embs, dim=0)

def load_methods_csv(csv_path):
    print("Loading METHODS_CSV...")
    df = pd.read_csv(csv_path)
    # method_signature -> info
    mapping = {}
    clean_sig = ""
    for _, row in df.iterrows():
        sig = str(row['method_signature'])
        if '(' in sig:
            clean_sig = sig.split('(')[0]
        else:
            clean_sig = sig
        mapping[clean_sig] = row.to_dict()
    print(f"Loaded {len(mapping)} methods from CSV.")
    return mapping

def load_enre_json(json_path):
    print("Loading ENRE_JSON...")
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
    
    print(f"Loaded {len(valid_nodes)} valid nodes and {len(seen_edges)} valid edges from ENRE_JSON.")
    return valid_nodes, qname_to_id, id_to_qname, adj, reverse_adj

def get_class_code(file_path, class_qname):
    full_path = os.path.join(PROJECT_PATH, file_path.lstrip('/'))
    if not os.path.exists(full_path):
        # Try without lstrip if it was already absolute or relative differently
        if os.path.exists(file_path):
            full_path = file_path
        else:
            return ""
            
    try:
        from tree_sitter import Language, Parser
        import tree_sitter_python as tspython
        
        PY_LANGUAGE = Language(tspython.language())
        parser = Parser(PY_LANGUAGE)
        
        with open(full_path, 'r') as f:
            content = f.read()
        
        tree = parser.parse(bytes(content, "utf8"))
        root_node = tree.root_node
        
        # Search for class definition matching class_qname
        # Note: class_qname might be fully qualified (e.g. module.submodule.ClassName)
        # But in the file, it is just "class ClassName..."
        # We need to extract the short class name.
        short_name = class_qname.split('.')[-1]
        
        # Simple DFS to find the class definition
        def find_class_node(node, target_name):
            if node.type == 'class_definition':
                # Find name node
                name_node = node.child_by_field_name('name')
                if name_node and content[name_node.start_byte:name_node.end_byte] == target_name:
                    return node
            
            for child in node.children:
                res = find_class_node(child, target_name)
                if res:
                    return res
            return None

        target_node = find_class_node(root_node, short_name)
        
        if target_node:
            return content[target_node.start_byte:target_node.end_byte]
        else:
            print(f"Class {short_name} not found in {full_path}")
            return ""

    except Exception as e:
        print(f"Error extracting class code from {full_path}: {e}")
        # Fallback: read file
        try:
            with open(full_path, 'r') as f:
                return f.read()
        except:
            return ""

def expand_graph(G, valid_nodes, adj, reverse_adj, method_map, id_to_qname):
    # G is the original graph from GML
    # Mark original nodes
    for node in G.nodes():
        G.nodes[node]['src_type'] = 'original'
        
    nodes_to_add = []
    edges_to_add = []
    
    # Iterate over original nodes to expand
    # We need to be careful not to modify G while iterating
    original_nodes = list(G.nodes(data=True))
    
    added_node_ids = set() # Track nodes added in this expansion step to avoid duplicates in nodes_to_add
    
    for node_id, node_data in original_nodes:
        # We need to map node_id (from GML) to ENRE id if they are different.
        # Assuming GML node ids match ENRE ids or we can look them up.
        # In build_graph.py, it seems we used ENRE ids.
        
        # Check if node exists in valid_nodes (ENRE data)
        # GML might store IDs as strings or ints, ENRE uses ints usually.
        try:
            enre_id = int(node_id)
        except:
            continue
            
        if enre_id not in valid_nodes:
            continue
            
        enre_node = valid_nodes[enre_id]
        category = node_data.get('category')
        
        if category == 'Function':
            # 1. Outgoing edges: Call, Use, Contain -> Function
            if enre_id in adj:
                for dest_id, kind in adj[enre_id]:
                    if kind in ['Call', 'Use', 'Contain']:
                        if dest_id in valid_nodes and valid_nodes[dest_id].get('category') == 'Function':
                            # check if it's in G first.
                            is_in_graph = str(dest_id) in G.nodes or dest_id in G.nodes
                            
                            if not is_in_graph:
                                if dest_id not in added_node_ids:
                                    # Prepare new node
                                    dest_node_info = valid_nodes[dest_id]
                                    qname = id_to_qname.get(dest_id, "")
                                    
                                    # Add node and edge
                                    nodes_to_add.append((dest_id, {
                                        'sig': qname,
                                        'category': 'Function',
                                        'src_type': 'expand',
                                        'method_signature': method_map[qname].get('method_signature', ''),
                                        'func_file': method_map[qname].get('func_file', ''),
                                        'method_code': method_map[qname].get('method_code', '')
                                    }))
                                    added_node_ids.add(dest_id)
                                    edges_to_add.append((node_id, dest_id, {'kind': kind}))
                            
                            
                            
            # 2. Incoming edges: Define -> Class
            if enre_id in reverse_adj:
                for src_id, kind in reverse_adj[enre_id]:
                    if kind == 'Define':
                        if src_id in valid_nodes and valid_nodes[src_id].get('category') == 'Class':
                             # Check if src_id is in graph
                            is_in_graph = str(src_id) in G.nodes or src_id in G.nodes
                            
                            if not is_in_graph:
                                if src_id not in added_node_ids:
                                    # Prepare new node
                                    src_node_info = valid_nodes[src_id]
                                    qname = id_to_qname.get(src_id, "")
                                    file_path = src_node_info['File']
                                    
                                    class_code = ""
                                    if file_path:
                                        class_code = get_class_code(file_path, qname)
                                        
                                    nodes_to_add.append((src_id, {
                                        'sig': qname,
                                        'category': 'Class',
                                        'src_type': 'expand',
                                        'method_signature': qname,
                                        'func_file': file_path,
                                        'method_code': class_code 
                                    }))
                                    added_node_ids.add(src_id)
                                    edges_to_add.append((src_id, node_id, {'kind': kind}))
                            
                            

    # Apply additions
    for n, attr in nodes_to_add:
        if n not in G:
            G.add_node(n, **attr)
        else:
            # If node exists (e.g. was original), maybe update src_type? 
            # User said "原图里面的点的src_type都为original". 
            # If we encounter it again as expansion target, we should probably NOT overwrite 'original' with 'expand'.
            # So only add if not present.
            pass
            
    for u, v, attr in edges_to_add:
        if not G.has_edge(u, v):
            G.add_edge(u, v, **attr)
            
    return G

def main():
    model, device = load_unixcoder_model()

    # 1. Load Data
    method_map = load_methods_csv(METHODS_CSV)
    valid_nodes, qname_to_id, id_to_qname, adj, reverse_adj = load_enre_json(ENRE_JSON)
    
    tasks = []
    print("Loading Tasks...")
    with open(FILTERED_PATH, 'r') as f:
        for line in f:
            if line.strip():
                tasks.append(json.loads(line))
    print(f"Loaded {len(tasks)} tasks.")
    
    # 2. Process each task
    for i, task in enumerate(tasks):
        task_id = task.get('example_id', i + 1)
        # Query construction
        query = task.get('requirement', {}).get('Functionality', '') + ' ' + task.get('requirement', {}).get('Arguments', '')
        
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
            
        print(f"Processing Task {task_id}: Graph has {len(G.nodes)} nodes and {len(G.edges)} edges.")
        
        # Expand Graph
        expanded_G = expand_graph(G, valid_nodes, adj, reverse_adj, method_map, id_to_qname)
        print(f"  -> Expanded: {len(expanded_G.nodes)} nodes and {len(expanded_G.edges)} edges.")

        # Save Expanded Graph (Mid)
        mid_gml_filename = f"task_{task_id}_mid.gml"
        mid_gml_path = os.path.join(OUTPUT_GRAPH_PATH, mid_gml_filename)
        nx.write_gml(expanded_G, mid_gml_path)
        print(f"  -> Saved expanded graph to {mid_gml_path}")

        # 输出一些调试信息，比如前3个任务，各自输出扩展节点中的前3个
        if i < 3:
            print(f"  -> Debug: First 3 expanded nodes for task {task_id}:")
            for n in list(expanded_G.nodes(data=True))[:10]:
                print(f"    - Node ID: {n[0]}, Attributes: {n[1]}")

        
        # Compute similarity scores
        # 1. Encode query
        nl_emb = encode_nl_with_unixcoder(model, device, query) # (1, D)
        
        # 2. Encode nodes (method_code)
        # We need to collect codes for all nodes in expanded_G
        node_list = list(expanded_G.nodes(data=True))
        node_codes = []
        node_ids = []
        
        for n, d in node_list:
            # Try to get method_code
            # For original nodes (Function), we might need to look up in method_map if not present
            # or check ENRE info.
            # In build_graph.py, we didn't explicitly store method_code in GML nodes?
            # Let's check if G nodes have 'method_code' or 'qualifiedName' to look up.
            
            code = ""
            if 'method_code' in d:
                code = d['method_code']
            
            node_codes.append(code if code else "")
            node_ids.append(n)
            
        # Encode all codes
        # print("  -> Encoding node codes...")
        if node_codes:
             code_embs = encode_code_with_unixcoder(model, device, node_codes, batch_size=32) # (N, D)
             
             # Calculate cosine similarity
             # nl_emb: (1, D), code_embs: (N, D)
             # sims: (1, N)
             if code_embs.shape[0] > 0:
                 sims = torch.mm(nl_emb.to(device), code_embs.to(device).t()).squeeze(0).cpu().numpy() # (N,)
             else:
                 sims = np.zeros(len(node_codes))
        else:
            sims = np.zeros(len(node_codes))
            
        # Assign personalization vector
        personalization = {}
        for idx, n in enumerate(node_ids):
            # Ensure non-negative
            score = max(0.0, float(sims[idx]))
            personalization[n] = score
            
        # Normalize personalization (PageRank requires sum=1, but nx handles it if sum>0)
        # If all zero, use uniform? Or just uniform.
        total_score = sum(personalization.values())
        if total_score == 0:
            print("???")
            personalization = {n: 1.0/len(node_ids) for n in node_ids}
            
        # Run Personalized PageRank
        try:
            # alpha=0.85 is standard, but we can tune.
            ppr_scores = nx.pagerank(expanded_G, alpha=0.85, personalization=personalization)
        except Exception as e:
            print(f"  -> PageRank failed: {e}. Using similarity scores directly.")
            ppr_scores = personalization
            
        # Select Top-K
        K = 15
        sorted_nodes = sorted(ppr_scores.items(), key=lambda x: x[1], reverse=True)
        top_k_nodes = [n for n, s in sorted_nodes[:K]]
        
        # Create Subgraph
        subgraph = expanded_G.subgraph(top_k_nodes).copy()

        # Add score to nodes
        for node in subgraph.nodes():
            if node in ppr_scores:
                subgraph.nodes[node]['score'] = ppr_scores[node]
        
        # Save Subgraph
        rank_gml_filename = f"task_{task_id}_rank.gml"
        rank_gml_path = os.path.join(OUTPUT_GRAPH_PATH, rank_gml_filename)
        nx.write_gml(subgraph, rank_gml_path)
        print(f"  -> Saved top-{K} subgraph to {rank_gml_path}")

if __name__ == "__main__":
    main()
