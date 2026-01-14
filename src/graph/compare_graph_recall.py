import os
import glob
import re
import json
import networkx as nx
import pandas as pd
from collections import defaultdict

# Configuration
GRAPH_RESULTS_DIR = '/data/data_public/riverbag/testRepoSummaryOut/alembic/1:3/graph_results'
FILTERED_JSONL_PATH = '/data/data_public/riverbag/testRepoSummaryOut/alembic/1:3/filtered.jsonl'
OUTPUT_REPORT_FILE = '/data/data_public/riverbag/testRepoSummaryOut/alembic/1:3/alembic_match_comparison_report.csv'

def load_ground_truth(task_id):
    """
    Loads ground truth methods for a specific task ID from the JSONL file.
    Returns a set of ground truth signatures.
    """
    gt_sigs = set()
    if not os.path.exists(FILTERED_JSONL_PATH):
        print(f"Warning: Filtered JSONL file not found at {FILTERED_JSONL_PATH}")
        return gt_sigs

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
                    gt_sigs.update(dependency.get('intra_class', []))
                    gt_sigs.update(dependency.get('intra_file', []))
                    gt_sigs.update(dependency.get('cross_file', []))
                    break
    except ValueError:
        print(f"Error: Invalid task_id '{task_id}' - must be an integer.")
    except Exception as e:
        print(f"Error reading filtered JSONL: {e}")
        
    return gt_sigs

def get_matched_count(gml_path, gt_sigs):
    """
    Reads a GML file and counts how many of its nodes match the Ground Truth signatures.
    """
    if not os.path.exists(gml_path):
        return 0
    
    try:
        G = nx.read_gml(gml_path)
    except Exception as e:
        print(f"Error reading {gml_path}: {e}")
        return 0
        
    matched_count = 0
    for node_id in G.nodes():
        node = G.nodes[node_id]
        node_sig = node.get('sig')
        
        if node_sig:
             # Normalize logic matching visualize_gml.py
             normalized_sig = node_sig
             if '.' in node_sig:
                 parts = node_sig.split('.')
                 if len(parts) > 1:
                     normalized_sig = '.'.join(parts[1:])
             
             if normalized_sig in gt_sigs:
                 matched_count += 1
                 
    return matched_count

def main():
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
    
    print(f"{'Task ID':<10} | {'GT Total':<10} | {'Ori Match':<10} | {'Mid Match':<10} | {'Rank Match':<10} | {'Improvement (Mid)':<20} | {'Improvement (Rank)':<20}")
    print("-" * 110)

    for task_id in sorted_task_ids:
        types = task_files[task_id]
        
        # Load GT
        gt_sigs = load_ground_truth(task_id)
        gt_total = len(gt_sigs)
        
        # Calculate matches
        ori_match = get_matched_count(types.get('ori'), gt_sigs)
        mid_match = get_matched_count(types.get('mid'), gt_sigs)
        rank_match = get_matched_count(types.get('rank'), gt_sigs)
        
        mid_improv = mid_match - ori_match
        rank_improv = rank_match - ori_match
        
        print(f"{task_id:<10} | {gt_total:<10} | {ori_match:<10} | {mid_match:<10} | {rank_match:<10} | {mid_improv:<+20} | {rank_improv:<+20}")
        
        results.append({
            'task_id': task_id,
            'gt_total': gt_total,
            'ori_match': ori_match,
            'mid_match': mid_match,
            'rank_match': rank_match,
            'mid_improvement': mid_improv,
            'rank_improvement': rank_improv
        })

    # 3. Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_REPORT_FILE, index=False)
    print(f"\nReport saved to: {OUTPUT_REPORT_FILE}")

    # 4. Statistics Summary
    total_tasks = len(results)
    if total_tasks > 0:
        # Mid Stats
        mid_improved = sum(1 for r in results if r['mid_match'] > r['ori_match'])
        mid_decreased = sum(1 for r in results if r['mid_match'] < r['ori_match'])
        
        # Rank Stats
        rank_improved = sum(1 for r in results if r['rank_match'] > r['ori_match'])
        rank_decreased = sum(1 for r in results if r['rank_match'] < r['ori_match'])
        
        # Totals
        sum_gt = sum(r['gt_total'] for r in results)
        sum_ori = sum(r['ori_match'] for r in results)
        sum_mid = sum(r['mid_match'] for r in results)
        sum_rank = sum(r['rank_match'] for r in results)
        
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
