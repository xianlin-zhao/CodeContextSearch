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
import sys
sys.path.append("/data/data_public/riverbag/CodeContextSearch/src")

from search.search_models.unixcoder import UniXcoder

# METHODS_CSV = "/data/data_public/riverbag/testRepoSummaryOut/Filited/boto/methods.csv"
# ENRE_JSON = "/data/data_public/riverbag/testRepoSummaryOut/Filited/boto/boto-report-enre.json"
# FILTERED_PATH = "/data/data_public/riverbag/testRepoSummaryOut/Filited/boto/filtered.jsonl"
# OUTPUT_GRAPH_PATH = "/data/zxl/Search2026/outputData/devEvalSearchOut/Internet_boto/0115/graph_results"
# PROJECT_PATH = "/data/lowcode_public/DevEval/Source_Code/Internet"

METHODS_CSV = "/data/data_public/riverbag/testRepoSummaryOut/211/alembic/methods.csv"
ENRE_JSON = "/data/data_public/riverbag/testRepoSummaryOut/211/alembic/alembic-report-enre.json"
FILTERED_PATH = "/data/data_public/riverbag/testRepoSummaryOut/211/alembic/filtered.jsonl"
OUTPUT_GRAPH_PATH = "/data/data_public/riverbag/testRepoSummaryOut/211/alembic/graph_results"
# PROJECT_PATH = "/data/lowcode_public/DevEval/Source_Code/System" #mrjob
# PROJECT_PATH = "/data/lowcode_public/DevEval/Source_Code/Internet" #boto
PROJECT_PATH = "/data/lowcode_public/DevEval/Source_Code/Database" #alembic
TOP_KS = [15]

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
    for node in G.nodes():
        if 'src_type' not in G.nodes[node]:
            G.nodes[node]['src_type'] = 'original'

    def preferred_id_type():
        for n in G.nodes():
            return str if isinstance(n, str) else int
        return str

    preferred_type = preferred_id_type()

    def node_key_for_enre_id(enre_id):
        if enre_id in G.nodes:
            return enre_id
        sid = str(enre_id)
        if sid in G.nodes:
            return sid
        return sid if preferred_type is str else enre_id

    def safe_int(x):
        try:
            return int(x)
        except Exception:
            return None

    def get_method_info(qname):
        if not qname:
            return {}
        if qname in method_map:
            return method_map[qname]
        base = qname.split('(')[0]
        if base in method_map:
            return method_map[base]
        short = base.split('.')[-1]
        return method_map.get(short, {})

    nodes_to_add = []
    edges_to_add = []
    added_node_keys = set()

    def ensure_function_node(enre_id):
        key = node_key_for_enre_id(enre_id)
        if key in G.nodes or key in added_node_keys:
            return key
        qname = id_to_qname.get(enre_id, "")
        method_info = get_method_info(qname)
        nodes_to_add.append((key, {
            'sig': qname,
            'category': 'Function',
            'src_type': 'expand',
            'method_signature': method_info.get('method_signature', ''),
            'func_file': method_info.get('func_file', ''),
            'method_code': method_info.get('method_code', '')
        }))
        added_node_keys.add(key)
        return key

    original_function_enre_ids = []
    for node_id, node_data in list(G.nodes(data=True)):
        if node_data.get('src_type') != 'original' or node_data.get('category') != 'Function':
            continue
        enre_id = safe_int(node_id)
        if enre_id is None or enre_id not in valid_nodes:
            continue
        original_function_enre_ids.append(enre_id)

    for enre_id in original_function_enre_ids:
        src_key = node_key_for_enre_id(enre_id)

        if enre_id in adj:
            for dest_id, kind in adj[enre_id]:
                if kind not in ['Call', 'Use', 'Contain']:
                    continue
                if dest_id not in valid_nodes or valid_nodes[dest_id].get('category') != 'Function':
                    continue
                dest_key = ensure_function_node(dest_id)
                edges_to_add.append((src_key, dest_key, {'kind': kind}))

        if enre_id in reverse_adj:
            for caller_id, kind in reverse_adj[enre_id]:
                if kind != 'Call':
                    continue
                if caller_id not in valid_nodes or valid_nodes[caller_id].get('category') != 'Function':
                    continue
                caller_key = ensure_function_node(caller_id)
                edges_to_add.append((caller_key, src_key, {'kind': kind}))

    for n, attr in nodes_to_add:
        if n not in G:
            G.add_node(n, **attr)

    for u, v, attr in edges_to_add:
        if not G.has_edge(u, v):
            G.add_edge(u, v, **attr)

    all_function_enre_ids = set()
    for node_id, node_data in G.nodes(data=True):
        if node_data.get('category') != 'Function':
            continue
        enre_id = safe_int(node_id)
        if enre_id is None or enre_id not in valid_nodes:
            continue
        all_function_enre_ids.add(enre_id)

    class_nodes_to_add = []
    class_edges_to_add = []
    class_added_keys = set()

    def ensure_class_node_in_second_pass(enre_id):
        key = node_key_for_enre_id(enre_id)
        if key in G.nodes or key in class_added_keys:
            return key
        src_node_info = valid_nodes[enre_id]
        qname = id_to_qname.get(enre_id, "")
        file_path = src_node_info.get('File', "")
        class_code = get_class_code(file_path, qname) if file_path else ""
        class_nodes_to_add.append((key, {
            'sig': qname,
            'category': 'Class',
            'src_type': 'expand',
            'method_signature': qname,
            'func_file': file_path,
            'method_code': class_code
        }))
        class_added_keys.add(key)
        return key

    for func_enre_id in all_function_enre_ids:
        func_key = node_key_for_enre_id(func_enre_id)
        if func_enre_id not in reverse_adj:
            continue
        for class_id, kind in reverse_adj[func_enre_id]:
            if kind != 'Define':
                continue
            if class_id not in valid_nodes or valid_nodes[class_id].get('category') != 'Class':
                continue
            class_key = ensure_class_node_in_second_pass(class_id)
            class_edges_to_add.append((class_key, func_key, {'kind': kind}))

    for n, attr in class_nodes_to_add:
        if n not in G:
            G.add_node(n, **attr)

    for u, v, attr in class_edges_to_add:
        if not G.has_edge(u, v):
            G.add_edge(u, v, **attr)

    return G

def get_graph_input_dirs(base_dir):
    try:
        children = [os.path.join(base_dir, name) for name in os.listdir(base_dir)]
    except FileNotFoundError:
        return []

    def has_ori_gml_files(dir_path):
        try:
            for name in os.listdir(dir_path):
                if name.endswith("_ori.gml"):
                    return True
        except Exception:
            return False
        return False

    subdirs = [p for p in children if os.path.isdir(p)]
    subdirs_with_ori = [p for p in subdirs if has_ori_gml_files(p)]
    if subdirs_with_ori:
        return sorted(subdirs_with_ori)

    if has_ori_gml_files(base_dir):
        return [base_dir]

    return sorted(subdirs) if subdirs else [base_dir]

def process_graph_dir(graph_dir, tasks, model, device, method_map, valid_nodes, id_to_qname, adj, reverse_adj):
    def is_truthy(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return v != 0
        if isinstance(v, str):
            return v.strip().lower() in {"true", "1", "yes", "y"}
        return False

    for i, task in enumerate(tasks):
        task_id = task.get('example_id', i + 1)
        query = task.get('requirement', {}).get('Functionality', '') + ' ' + task.get('requirement', {}).get('Arguments', '')

        dependency = task.get('dependency', {})
        gt_methods = []
        gt_methods.extend(dependency.get('intra_class', []))
        gt_methods.extend(dependency.get('intra_file', []))
        gt_methods.extend(dependency.get('cross_file', []))

        gml_filename = f"task_{task_id}_ori.gml"
        gml_path = os.path.join(graph_dir, gml_filename)

        if not os.path.exists(gml_path):
            print(f"Graph file not found for task {task_id}: {gml_path}")
            continue

        try:
            G = nx.read_gml(gml_path)
        except Exception as e:
            print(f"Error reading GML for task {task_id}: {e}")
            continue

        original_node_ids = set(G.nodes())

        print(f"Processing Task {task_id}: Graph has {len(G.nodes)} nodes and {len(G.edges)} edges.")

        expanded_G = expand_graph(G, valid_nodes, adj, reverse_adj, method_map, id_to_qname)
        print(f"  -> Expanded: {len(expanded_G.nodes)} nodes and {len(expanded_G.edges)} edges.")

        mid_gml_filename = f"task_{task_id}_mid.gml"
        mid_gml_path = os.path.join(graph_dir, mid_gml_filename)
        nx.write_gml(expanded_G, mid_gml_path)
        print(f"  -> Saved expanded graph to {mid_gml_path}")

        # if i < 3:
        #     print(f"  -> Debug: First 3 expanded nodes for task {task_id}:")
        #     for n in list(expanded_G.nodes(data=True))[:10]:
        #         print(f"    - Node ID: {n[0]}, Attributes: {n[1]}")

        nl_emb = encode_nl_with_unixcoder(model, device, query)

        node_list = list(expanded_G.nodes(data=True))
        node_codes = []
        node_ids = []

        for n, d in node_list:
            code = ""
            if 'method_code' in d:
                code = d['method_code']

            node_codes.append(code if code else "")
            node_ids.append(n)

        if node_codes:
            code_embs = encode_code_with_unixcoder(model, device, node_codes, batch_size=32)

            if code_embs.shape[0] > 0:
                sims = torch.mm(nl_emb.to(device), code_embs.to(device).t()).squeeze(0).cpu().numpy()
            else:
                sims = np.zeros(len(node_codes))
        else:
            sims = np.zeros(len(node_codes))

        personalization = {}
        for idx, n in enumerate(node_ids):
            score = max(0.0, float(sims[idx]))
            personalization[n] = score

        boosted_nodes = 0
        boosted_total_delta = 0.0
        for n in original_node_ids:
            attrs = expanded_G.nodes.get(n, {})
            inc = 0.0
            if is_truthy(attrs.get("is_SameFeature")):
                inc += 0.1
            if is_truthy(attrs.get("is_SameFile")):
                inc += 0.1
            if is_truthy(attrs.get("is_SimilarMethod")):
                inc += 0.1
            if inc <= 0:
                continue

            neighbors = set()
            try:
                neighbors.update(expanded_G.successors(n))
            except Exception:
                pass
            try:
                neighbors.update(expanded_G.predecessors(n))
            except Exception:
                pass

            for nb in neighbors:
                if nb in original_node_ids:
                    continue
                personalization[nb] = float(personalization.get(nb, 0.0)) + inc
                boosted_nodes += 1
                boosted_total_delta += inc

        if boosted_nodes > 0:
            print(
                f"  -> Personalization boosted: edges_to_expand_nodes={boosted_nodes} total_delta={boosted_total_delta:.4f}",
                flush=True,
            )

        total_score = sum(personalization.values())
        if total_score == 0:
            print("???")
            personalization = {n: 1.0/len(node_ids) for n in node_ids}

        try:
            ppr_scores = nx.pagerank(expanded_G, alpha=0.85, personalization=personalization)
        except Exception as e:
            print(f"  -> PageRank failed: {e}. Using similarity scores directly.")
            ppr_scores = personalization

        top_ks_raw = TOP_KS if isinstance(TOP_KS, (list, tuple, set)) else [TOP_KS]
        top_ks = []
        for k in top_ks_raw:
            try:
                k_int = int(k)
            except Exception:
                continue
            if k_int > 0:
                top_ks.append(k_int)
        top_ks = list(dict.fromkeys(top_ks))
        if not top_ks:
            top_ks = [15]

        sorted_nodes = sorted(ppr_scores.items(), key=lambda x: x[1], reverse=True)
        rank_gml_filename = f"task_{task_id}_rank.gml"

        for K in top_ks:
            top_k_nodes = [n for n, _ in sorted_nodes[:K]]

            subgraph = expanded_G.subgraph(top_k_nodes).copy()

            for node in subgraph.nodes():
                if node in ppr_scores:
                    subgraph.nodes[node]['score'] = ppr_scores[node]

            top_k_dir = os.path.join(graph_dir, f"PageRank-{K}-subgraph")
            os.makedirs(top_k_dir, exist_ok=True)
            rank_gml_path = os.path.join(top_k_dir, rank_gml_filename)
            nx.write_gml(subgraph, rank_gml_path)
            print(f"  -> Saved top-{K} subgraph to {rank_gml_path}")

def main():
    model, device = load_unixcoder_model()
    os.makedirs(OUTPUT_GRAPH_PATH, exist_ok=True)

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
    
    graph_dirs = get_graph_input_dirs(OUTPUT_GRAPH_PATH)
    for graph_dir in graph_dirs:
        if not os.path.isdir(graph_dir):
            continue
        print(f"Processing graph directory: {graph_dir}")
        process_graph_dir(
            graph_dir=graph_dir,
            tasks=tasks,
            model=model,
            device=device,
            method_map=method_map,
            valid_nodes=valid_nodes,
            id_to_qname=id_to_qname,
            adj=adj,
            reverse_adj=reverse_adj,
        )

if __name__ == "__main__":
    main()
