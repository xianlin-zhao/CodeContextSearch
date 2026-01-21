import os
import glob
import re
import json
import networkx as nx
import pandas as pd
from collections import defaultdict

# Configuration
GRAPH_RESULTS_DIR = '/data/data_public/riverbag/testRepoSummaryOut/Filited/boto/expand_graph_results/top-20-subgraph'
FILTERED_JSONL_PATH = '/data/data_public/riverbag/testRepoSummaryOut/Filited/boto/filtered.jsonl'
OUTPUT_REPORT_FILE = '/data/data_public/riverbag/testRepoSummaryOut/Filited/boto/duoduoduo_expand_graph_match_comparison_report.csv'
ENRE_JSON = '/data/data_public/riverbag/testRepoSummaryOut/Filited/boto/boto-report-enre.json'

variables_enre = set()

def load_enre_variables(json_path):
    if not os.path.exists(json_path):
        print(f"Warning: ENRE JSON file not found at {json_path}")
        return

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading ENRE JSON: {e}")
        return

    variables = data.get('variables', [])
    for var in variables:
        if not var.get('category', '').startswith('Un'):
            if var.get('category') == 'Variable':
                qname = var.get('qualifiedName')
                if qname:
                    variables_enre.add(qname)

def _normalize_symbol(s: str) -> str:
    if "(" in s:
        return s.split("(", 1)[0]
    return s

def compute_task_recall(dependency, searched_context_code_list):
    dep = dependency or []
    dep_set = {x for x in dep}
    retrieved_set = {
        _normalize_symbol(str(x.get("method_signature", "")))
        for x in searched_context_code_list
        if isinstance(x, dict)
    }
    dep_total = len(dep_set)
    hit = len(dep_set & retrieved_set) if dep_total > 0 else 0

    for x in dep:
        if x in variables_enre:
            var_name = x.split('.')[-1]
            for context_code in searched_context_code_list:
                code_detail = context_code.get("method_code") or ""
                if var_name in code_detail:
                    hit += 1
                    break

    recall = (hit / dep_total) if dep_total > 0 else None
    return {
        "dependency_total": dep_total,
        "dependency_hit": hit,
        "recall": recall,
    }

def load_ground_truth(task_id):
    """
    Loads ground truth dependency symbols for a specific task ID from the JSONL file.
    Returns a list of symbols.
    """
    dep = []
    if not os.path.exists(FILTERED_JSONL_PATH):
        print(f"Warning: Filtered JSONL file not found at {FILTERED_JSONL_PATH}")
        return dep

    try:
        target_line_num = int(task_id)
        current_line_num = 0
        with open(FILTERED_JSONL_PATH, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                current_line_num += 1
                
                # Check if this is the target line (1-based index)
                if current_line_num == target_line_num:
                    data = json.loads(line)
                    dependency = data.get('dependency', {})
                    dep.extend(dependency.get('intra_class', []))
                    dep.extend(dependency.get('intra_file', []))
                    dep.extend(dependency.get('cross_file', []))
                    break
    except ValueError:
        print(f"Error: Invalid task_id '{task_id}' - must be an integer.")
    except Exception as e:
        print(f"Error reading filtered JSONL: {e}")
        
    return dep

def load_context_code_list_from_gml(gml_path):
    if not gml_path or not os.path.exists(gml_path):
        return []

    try:
        G = nx.read_gml(gml_path)
    except Exception as e:
        print(f"Error reading {gml_path}: {e}")
        return []

    context_code_list = []
    for node_id in G.nodes():
        node = G.nodes[node_id]
        if node.get('category') != 'Function':
            continue

        context_code_list.append({
            "method_signature": str(node.get("method_signature", "")),
            "method_code": str(node.get("method_code", "")),
        })

    return context_code_list

def get_matched_count(gml_path, gt_sigs):
    """
    Reads a GML file and computes dependency hit count with variable-aware matching.
    """
    context_code_list = load_context_code_list_from_gml(gml_path)
    stats = compute_task_recall(list(gt_sigs), context_code_list)
    return stats.get("dependency_hit", 0)

def list_rank_subdirs(base_dir):
    if not base_dir or not os.path.isdir(base_dir):
        return []

    rank_dirs = []
    for name in os.listdir(base_dir):
        full = os.path.join(base_dir, name)
        if not os.path.isdir(full):
            continue
        if name.startswith("PageRank-") and name.endswith("-subgraph"):
            rank_dirs.append(name)
    return sorted(rank_dirs)

def sanitize_col(s):
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(s)).strip("_")

def main():
    load_enre_variables(ENRE_JSON)

    # 1. Group files by task_id
    # Pattern: task_{id}_{type}.gml
    base_files = glob.glob(os.path.join(GRAPH_RESULTS_DIR, "*.gml"))
    task_files = defaultdict(dict)

    for f in base_files:
        basename = os.path.basename(f)
        match = re.search(r'task_(\d+)_(\w+)\.gml', basename)
        if match:
            task_id = match.group(1)
            file_type = match.group(2)
            task_files[task_id][file_type] = f

    rank_subdirs = list_rank_subdirs(GRAPH_RESULTS_DIR)
    rank_files_by_dir = {name: {} for name in rank_subdirs}
    for name in rank_subdirs:
        rank_dir = os.path.join(GRAPH_RESULTS_DIR, name)
        for f in glob.glob(os.path.join(rank_dir, "*.gml")):
            basename = os.path.basename(f)
            match = re.search(r'task_(\d+)_rank\.gml', basename)
            if match:
                task_id = match.group(1)
                rank_files_by_dir[name][task_id] = f

    # 2. Analyze each task
    results = []
    
    # Sort by task_id integer
    sorted_task_ids = sorted(task_files.keys(), key=lambda x: int(x))
    
    header_parts = [
        f"{'Task ID':<10}",
        f"{'GT Total':<10}",
        f"{'Ori Hit':<10}",
        f"{'Mid Hit':<10}",
        f"{'Ori Recall':<10}",
        f"{'Mid Recall':<10}",
    ]
    for name in rank_subdirs:
        header_parts.append(f"{name + ' Hit':<18}")
        header_parts.append(f"{name + ' Recall':<18}")

    print(" | ".join(header_parts))
    print("-" * (len(" | ".join(header_parts))))

    for task_id in sorted_task_ids:
        types = task_files[task_id]
        
        # Load GT
        dep = load_ground_truth(task_id)
        gt_total = len(set(dep))

        ori_ctx = load_context_code_list_from_gml(types.get('ori'))
        mid_ctx = load_context_code_list_from_gml(types.get('mid'))

        ori_stats = compute_task_recall(dep, ori_ctx)
        mid_stats = compute_task_recall(dep, mid_ctx)

        ori_hit = ori_stats.get("dependency_hit", 0)
        mid_hit = mid_stats.get("dependency_hit", 0)

        ori_recall = ori_stats.get("recall")
        mid_recall = mid_stats.get("recall")

        def fmt_recall(x):
            return f"{x:.3f}" if isinstance(x, (int, float)) else "None"

        row_parts = [
            f"{task_id:<10}",
            f"{gt_total:<10}",
            f"{ori_hit:<10}",
            f"{mid_hit:<10}",
            f"{fmt_recall(ori_recall):<10}",
            f"{fmt_recall(mid_recall):<10}",
        ]
        rank_stats_by_dir = {}
        for name in rank_subdirs:
            rank_path = rank_files_by_dir.get(name, {}).get(task_id)
            rank_ctx = load_context_code_list_from_gml(rank_path)
            rank_stats = compute_task_recall(dep, rank_ctx)
            rank_stats_by_dir[name] = rank_stats
            rank_hit = rank_stats.get("dependency_hit", 0)
            rank_recall = rank_stats.get("recall")
            row_parts.append(f"{rank_hit:<18}")
            row_parts.append(f"{fmt_recall(rank_recall):<18}")

        print(" | ".join(row_parts))
        
        row = {
            'task_id': task_id,
            'dependency_total': gt_total,
            'ori_hit': ori_hit,
            'mid_hit': mid_hit,
            'ori_recall': ori_recall,
            'mid_recall': mid_recall,
        }
        for name in rank_subdirs:
            col = sanitize_col(name)
            stats = rank_stats_by_dir.get(name, {})
            row[f"rank_hit__{col}"] = stats.get("dependency_hit", 0)
            row[f"rank_recall__{col}"] = stats.get("recall")
        results.append(row)

    # 3. Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_REPORT_FILE, index=False)
    print(f"\nReport saved to: {OUTPUT_REPORT_FILE}")

    # 4. Statistics Summary
    total_tasks = len(results)
    if total_tasks > 0:
        def is_valid_recall(x):
            return isinstance(x, (int, float))

        # Mid Stats
        mid_improved = sum(
            1 for r in results
            if is_valid_recall(r.get('mid_recall')) and is_valid_recall(r.get('ori_recall')) and r['mid_recall'] > r['ori_recall']
        )
        mid_decreased = sum(
            1 for r in results
            if is_valid_recall(r.get('mid_recall')) and is_valid_recall(r.get('ori_recall')) and r['mid_recall'] < r['ori_recall']
        )

        # Totals
        sum_gt = sum(r['dependency_total'] for r in results)
        sum_ori = sum(r['ori_hit'] for r in results)
        sum_mid = sum(r['mid_hit'] for r in results)
        
        # Calculate percentages
        def calc_pct(count, total):
            return (count / total * 100) if total > 0 else 0
            
        print("\n" + "="*60)
        print("STATISTICS SUMMARY")
        print("="*60)
        
        print(f"Total Tasks: {total_tasks}")
        print("-" * 30)
        
        print(f"Mid Method:")
        print(f"  Improved:  {mid_improved:3d} tasks ({calc_pct(mid_improved, total_tasks):6.2f}%)")
        print(f"  Decreased: {mid_decreased:3d} tasks ({calc_pct(mid_decreased, total_tasks):6.2f}%)")
        
        for name in rank_subdirs:
            col = sanitize_col(name)
            improved = sum(
                1 for r in results
                if is_valid_recall(r.get(f"rank_recall__{col}")) and is_valid_recall(r.get('ori_recall')) and r[f"rank_recall__{col}"] > r['ori_recall']
            )
            decreased = sum(
                1 for r in results
                if is_valid_recall(r.get(f"rank_recall__{col}")) and is_valid_recall(r.get('ori_recall')) and r[f"rank_recall__{col}"] < r['ori_recall']
            )
            print("-" * 30)
            print(f"Rank Method ({name}):")
            print(f"  Improved:  {improved:3d} tasks ({calc_pct(improved, total_tasks):6.2f}%)")
            print(f"  Decreased: {decreased:3d} tasks ({calc_pct(decreased, total_tasks):6.2f}%)")
        
        print("="*60)
        print("RECALL STATISTICS")
        print("-" * 30)
        
        recall_ori = calc_pct(sum_ori, sum_gt)
        recall_mid = calc_pct(sum_mid, sum_gt)
        
        print(f"Total Ground Truth: {sum_gt}")
        print(f"Total Ori Matches:  {sum_ori} (Recall: {recall_ori:.2f}%)")
        print(f"Total Mid Matches:  {sum_mid} (Recall: {recall_mid:.2f}%)")
        for name in rank_subdirs:
            col = sanitize_col(name)
            sum_rank = sum(r.get(f"rank_hit__{col}", 0) for r in results)
            recall_rank = calc_pct(sum_rank, sum_gt)
            print(f"Total Rank Matches ({name}): {sum_rank} (Recall: {recall_rank:.2f}%)")
        
        print("-" * 30)
        print(f"Mid Recall Growth:  {recall_mid - recall_ori:+.2f}%")
        for name in rank_subdirs:
            col = sanitize_col(name)
            sum_rank = sum(r.get(f"rank_hit__{col}", 0) for r in results)
            recall_rank = calc_pct(sum_rank, sum_gt)
            print(f"Rank Recall Growth ({name}): {recall_rank - recall_ori:+.2f}%")
        
        print("="*60)
        print("AVERAGE MATCH STATISTICS")
        print("-" * 30)
        
        avg_ori = sum_ori / total_tasks
        avg_mid = sum_mid / total_tasks
        
        print(f"Average Ori Matches:  {avg_ori:.2f}")
        print(f"Average Mid Matches:  {avg_mid:.2f}")
        for name in rank_subdirs:
            col = sanitize_col(name)
            sum_rank = sum(r.get(f"rank_hit__{col}", 0) for r in results)
            avg_rank = sum_rank / total_tasks
            print(f"Average Rank Matches ({name}): {avg_rank:.2f}")
        print("="*60)

if __name__ == "__main__":
    main()
