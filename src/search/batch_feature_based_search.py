import argparse
import os
from typing import Any, Dict, List

import pandas as pd

from feature_based_search import analyze_project, TOP_SM, CLUSTER_KS, USE_REFINED_QUERY


EXCEL_PATH = "/data/zxl/Search2026/CodeContextSearch/src/generation/dev_eval/project_to_run/0311_5projects.xlsx"
SOURCE_CODE_DIR = "/data/lowcode_public/DevEval/Source_Code"
DEFAULT_BASE_SEARCH_OUT = "/data/zxl/Search2026/outputData/devEvalSearchOut/0316_batch_workflow"
DEFAULT_BASE_ENRE = "/data/zxl/Search2026/outputData/devEvalSearchOut/0316_batch_workflow"
DEFAULT_OUTPUT_CSV = "/data/zxl/Search2026/outputData/devEvalSearchOut/0316_batch_workflow/feature_batch_metrics.csv"
DATA_JSONL = "/data/zxl/Search2026/DevEval/data.jsonl"


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    for c in df.columns:
        c_str = str(c).strip()
        if c_str in ("项目名称", "project_name"):
            col_map[c] = "project_name"
        elif c_str in ("项目根目录", "project_root", "GRAPH_PROJECT_PATH"):
            col_map[c] = "project_root"
        elif c_str in ("ENRE路径", "enre_json"):
            col_map[c] = "enre_json"
    return df.rename(columns=col_map)


def _default_enre_path(project_name: str, base_enre: str) -> str:
    return os.path.join(base_enre, project_name, f"{project_name}-report-enre.json")


def _project_dir_from_project_root(project_root: str, source_code_dir: str) -> str:
    """
    从 project_root 中截取 PROJECT_DIR（Source_Code 后面的两段路径）。

    例如：
    project_root = /data/lowcode_public/DevEval/Source_Code/System/mrjob/mrjob
    PROJECT_DIR = System/mrjob
    """
    norm_root = os.path.normpath(project_root)
    norm_source = os.path.normpath(source_code_dir)
    parts = norm_root.split(os.sep)
    try:
        idx = parts.index(os.path.basename(norm_source))
    except ValueError:
        # 回退策略：直接取倒数三段中的中间两段
        if len(parts) >= 3:
            return "/".join(parts[-3:-1])
        if len(parts) >= 2:
            return "/".join(parts[-2:])
        return parts[-1]

    # 取 Source_Code 之后的两段
    start = idx + 1
    end = start + 2
    sub_parts = parts[start:end]
    return "/".join(sub_parts)


def run_one_project(
    project_name: str,
    project_root: str,
    enre_json: str,
    base_search_out: str,
    source_code_dir: str,
    data_jsonl: str,
) -> Dict[str, Any]:
    project_dir = _project_dir_from_project_root(project_root, source_code_dir)

    project_base = os.path.join(base_search_out, project_name)
    feature_csv = os.path.join(project_base, "features.csv")
    methods_csv = os.path.join(project_base, "methods.csv")
    filtered_path = os.path.join(project_base, "filtered.jsonl")
    refined_queries_cache_path = os.path.join(project_base, "refined_queries.json")

    if not os.path.exists(feature_csv):
        print(f"[skip] {project_name}: features.csv not found at {feature_csv}")
        return {}
    if not os.path.exists(methods_csv):
        print(f"[skip] {project_name}: methods.csv not found at {methods_csv}")
        return {}

    print(f"[run] {project_name} PROJECT_DIR={project_dir} (feature-based)")
    metrics = analyze_project(
        project_dir=project_dir,
        feature_csv=feature_csv,
        methods_csv=methods_csv,
        filtered_path=filtered_path,
        refined_queries_cache_path=refined_queries_cache_path,
        enre_json=enre_json,
        data_jsonl=data_jsonl,
        top_sm=TOP_SM,
        cluster_ks=CLUSTER_KS,
        use_refined_query=USE_REFINED_QUERY,
    )
    metrics["project_name"] = project_name
    return metrics


def _append_rows(
    rows: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    project_name: str,
) -> None:
    type_metrics = metrics.get("feature", {})
    num_examples = metrics.get("num_examples", 0)
    for k, v in type_metrics.items():
        rows.append(
            {
                "project_name": project_name,
                "project_dir": metrics.get("project_dir", ""),
                "metric_type": "feature",
                "k": int(k),
                "P": v.get("P", 0.0),
                "R": v.get("R", 0.0),
                "F1": v.get("F1", 0.0),
                "match": v.get("match", 0),
                "pred": v.get("pred", 0),
                "gt": v.get("top_gt", 0),
                "tasks": num_examples,
            }
        )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Batch run feature-based search experiments from Excel project list"
    )
    p.add_argument(
        "--excel_path",
        default=EXCEL_PATH,
        help="Excel 路径，含项目名称、项目根目录，可选 ENRE路径",
    )
    p.add_argument(
        "--base_search_out",
        default=DEFAULT_BASE_SEARCH_OUT,
        help="features/methods/filtered 等所在根目录",
    )
    p.add_argument(
        "--base_enre",
        default=DEFAULT_BASE_ENRE,
        help="ENRE 根目录（未填 enre_json 时推导用）",
    )
    p.add_argument(
        "--source_code_dir",
        default=SOURCE_CODE_DIR,
        help="DevEval 源代码根目录（用于从 project_root 截取 PROJECT_DIR）",
    )
    p.add_argument(
        "--data_jsonl",
        default=DATA_JSONL,
        help="DevEval 完整数据集 jsonl 路径",
    )
    p.add_argument(
        "--output_csv",
        default=DEFAULT_OUTPUT_CSV,
        help="批量实验结果（逐项目 + 总体）输出 CSV 路径",
    )
    p.add_argument("--sheet_name", default=0, help="Excel 工作表名或索引")
    args = p.parse_args()

    df = pd.read_excel(args.excel_path, sheet_name=args.sheet_name)
    df = _normalize_column_names(df)

    if "project_name" not in df.columns or "project_root" not in df.columns:
        raise SystemExit(
            "Excel 需包含「项目名称」与「项目根目录」列（或 project_name / project_root）"
        )

    all_rows: List[Dict[str, Any]] = []
    per_project_rows: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        project_name = str(row["project_name"]).strip()
        project_root = str(row["project_root"]).strip()
        if not project_name or not project_root:
            continue
        enre_json = row.get("enre_json")
        if pd.isna(enre_json) or not str(enre_json).strip():
            enre_json = _default_enre_path(project_name, args.base_enre)
        else:
            enre_json = str(enre_json).strip()

        metrics = run_one_project(
            project_name=project_name,
            project_root=project_root,
            enre_json=enre_json,
            base_search_out=args.base_search_out,
            source_code_dir=args.source_code_dir,
            data_jsonl=args.data_jsonl,
        )
        if not metrics:
            continue

        _append_rows(per_project_rows, metrics, project_name)

    all_rows.extend(per_project_rows)

    # 计算总体指标：按项目的任务数加权
    if per_project_rows:
        import collections

        grouped: Dict[int, List[Dict[str, Any]]] = collections.defaultdict(list)
        for r in per_project_rows:
            key = int(r["k"])
            grouped[key].append(r)

        for k, rows_for_k in grouped.items():
            total_tasks = sum(r["tasks"] for r in rows_for_k)
            if total_tasks <= 0:
                continue
            P_total = sum(r["P"] * r["tasks"] for r in rows_for_k) / total_tasks
            R_total = sum(r["R"] * r["tasks"] for r in rows_for_k) / total_tasks
            F1_total = sum(r["F1"] * r["tasks"] for r in rows_for_k) / total_tasks
            match_total = sum(r["match"] for r in rows_for_k)
            pred_total = sum(r["pred"] for r in rows_for_k)
            gt_total = sum(r["gt"] for r in rows_for_k)
            all_rows.append(
                {
                    "project_name": "ALL",
                    "project_dir": "",
                    "metric_type": "feature",
                    "k": k,
                    "P": P_total,
                    "R": R_total,
                    "F1": F1_total,
                    "match": match_total,
                    "pred": pred_total,
                    "gt": gt_total,
                    "tasks": total_tasks,
                }
            )

    if all_rows:
        out_dir = os.path.dirname(args.output_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        out_df = pd.DataFrame(all_rows)
        out_df.to_csv(args.output_csv, index=False)
        print(f"Saved batch feature-based metrics to {args.output_csv}")
    else:
        print("No metrics produced; nothing to save.")


if __name__ == "__main__":
    main()

