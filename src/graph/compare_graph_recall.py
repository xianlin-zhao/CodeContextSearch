import os
import glob
import re
import json
import networkx as nx
import pandas as pd
from collections import defaultdict

# Configuration
GRAPH_RESULTS_DIR = '/data/zxl/Search2026/outputData/devEvalSearchOut/Database_alembic/0128/top-15-subgraph'
FILTERED_JSONL_PATH = '/data/data_public/riverbag/testRepoSummaryOut/Filited/alembic/filtered.jsonl'
OUTPUT_REPORT_FILE = '/data/zxl/Search2026/outputData/devEvalSearchOut/Database_alembic/0128/expand_graph_match_comparison_report.csv'
ENRE_JSON = '/data/data_public/riverbag/testRepoSummaryOut/Filited/alembic/alembic-report-enre.json'

DEBUG = True

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

    hit_set = dep_set & retrieved_set
    hit = len(hit_set) if dep_total > 0 else 0

    for x in dep:
        if x in variables_enre:
            var_name = x.split('.')[-1]
            for context_code in searched_context_code_list:
                code_detail = context_code.get("method_code") or ""
                if var_name in code_detail:
                    hit_set.add(x)
                    hit += 1
                    break
    
    if DEBUG:
        print(f"retrieved_set: {retrieved_set}")
        print(f"hit_set: {hit_set}")

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

def main():
    load_enre_variables(ENRE_JSON)

    # 1. Group files by task_id
    # Pattern: task_{id}_{type}.gml
    files = glob.glob(os.path.join(GRAPH_RESULTS_DIR, "*.gml"))
    task_files = defaultdict(dict)
    
    for f in files:
        basename = os.path.basename(f)
        match = re.search(r'task_(\d+)_(\w+)\.gml', basename)
        if match:
            task_id = match.group(1)
            file_type = match.group(2) # ori, mid, rank
            task_files[task_id][file_type] = f

    # 2. Analyze each task
    results = []
    
    # Sort by task_id integer
    sorted_task_ids = sorted(task_files.keys(), key=lambda x: int(x))
    
    print(f"{'Task ID':<10} | {'GT Total':<10} | {'Ori Hit':<10} | {'Mid Hit':<10} | {'Rank Hit':<10} | {'Ori Recall':<10} | {'Mid Recall':<10} | {'Rank Recall':<10}")
    print("-" * 110)

    for task_id in sorted_task_ids:
        types = task_files[task_id]
        
        # Load GT
        dep = load_ground_truth(task_id)
        print(f"task_id: {task_id}, dep: {dep}")
        gt_total = len(set(dep))

        ori_ctx = load_context_code_list_from_gml(types.get('ori'))
        mid_ctx = load_context_code_list_from_gml(types.get('mid'))
        rank_ctx = load_context_code_list_from_gml(types.get('rank'))

        ori_stats = compute_task_recall(dep, ori_ctx)
        mid_stats = compute_task_recall(dep, mid_ctx)
        rank_stats = compute_task_recall(dep, rank_ctx)

        ori_hit = ori_stats.get("dependency_hit", 0)
        mid_hit = mid_stats.get("dependency_hit", 0)
        rank_hit = rank_stats.get("dependency_hit", 0)

        ori_recall = ori_stats.get("recall")
        mid_recall = mid_stats.get("recall")
        rank_recall = rank_stats.get("recall")

        def fmt_recall(x):
            return f"{x:.3f}" if isinstance(x, (int, float)) else "None"

        print(
            f"{task_id:<10} | {gt_total:<10} | {ori_hit:<10} | {mid_hit:<10} | {rank_hit:<10} | "
            f"{fmt_recall(ori_recall):<10} | {fmt_recall(mid_recall):<10} | {fmt_recall(rank_recall):<10}"
        )
        
        results.append({
            'task_id': task_id,
            'dependency_total': gt_total,
            'ori_hit': ori_hit,
            'mid_hit': mid_hit,
            'rank_hit': rank_hit,
            'ori_recall': ori_recall,
            'mid_recall': mid_recall,
            'rank_recall': rank_recall,
        })

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
        
        # Rank Stats
        rank_improved = sum(
            1 for r in results
            if is_valid_recall(r.get('rank_recall')) and is_valid_recall(r.get('ori_recall')) and r['rank_recall'] > r['ori_recall']
        )
        rank_decreased = sum(
            1 for r in results
            if is_valid_recall(r.get('rank_recall')) and is_valid_recall(r.get('ori_recall')) and r['rank_recall'] < r['ori_recall']
        )
        
        # Totals
        sum_gt = sum(r['dependency_total'] for r in results)
        sum_ori = sum(r['ori_hit'] for r in results)
        sum_mid = sum(r['mid_hit'] for r in results)
        sum_rank = sum(r['rank_hit'] for r in results)
        
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
        
        print("-" * 30)
        
        print(f"Rank Method:")
        print(f"  Improved:  {rank_improved:3d} tasks ({calc_pct(rank_improved, total_tasks):6.2f}%)")
        print(f"  Decreased: {rank_decreased:3d} tasks ({calc_pct(rank_decreased, total_tasks):6.2f}%)")
        
        print("="*60)
        print("RECALL STATISTICS")
        print("-" * 30)
        
        recall_ori = calc_pct(sum_ori, sum_gt)
        recall_mid = calc_pct(sum_mid, sum_gt)
        recall_rank = calc_pct(sum_rank, sum_gt)
        
        print(f"Total Ground Truth: {sum_gt}")
        print(f"Total Ori Matches:  {sum_ori} (Recall: {recall_ori:.2f}%)")
        print(f"Total Mid Matches:  {sum_mid} (Recall: {recall_mid:.2f}%)")
        print(f"Total Rank Matches: {sum_rank} (Recall: {recall_rank:.2f}%)")
        
        print("-" * 30)
        print(f"Mid Recall Growth:  {recall_mid - recall_ori:+.2f}%")
        print(f"Rank Recall Growth: {recall_rank - recall_ori:+.2f}%")
        
        print("="*60)
        print("AVERAGE MATCH STATISTICS")
        print("-" * 30)
        
        avg_ori = sum_ori / total_tasks
        avg_mid = sum_mid / total_tasks
        avg_rank = sum_rank / total_tasks
        
        print(f"Average Ori Matches:  {avg_ori:.2f}")
        print(f"Average Mid Matches:  {avg_mid:.2f}")
        print(f"Average Rank Matches: {avg_rank:.2f}")
        print("="*60)

if __name__ == "__main__":
    main()
