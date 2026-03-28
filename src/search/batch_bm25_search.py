import argparse
import contextlib
import json
import os
import sys
import traceback
from typing import Any, Dict, List

import pandas as pd

from bm25_search import analyze_project


# EXCEL_PATH = "/data/zxl/Search2026/CodeContextSearch/src/generation/dev_eval/project_to_run/0311_5projects.xlsx"
# SOURCE_CODE_DIR = "/data/zxl/Search2026/Datasets/Source_Code"
# DEFAULT_BASE_SEARCH_OUT = "/data/zxl/Search2026/outputData/devEvalSearchOut/0316_batch_workflow"
# DEFAULT_BASE_ENRE = "/data/zxl/Search2026/outputData/devEvalSearchOut/0316_batch_workflow"
# DEFAULT_OUTPUT_CSV = "/data/zxl/Search2026/outputData/devEvalSearchOut/0316_batch_workflow/bm25_batch_metrics.csv"
# DATA_JSONL = "/data/zxl/Search2026/DevEval/data.jsonl"

EXCEL_PATH = "/data/data_public/riverbag/CodeContextSearch/docs/deveval_project_path_stats.xlsx"
SOURCE_CODE_DIR = "/data/lowcode_public/DevEval_no_tests/Source_Code"
DEFAULT_BASE_SEARCH_OUT = "/data/data_public/riverbag/testRepoSummaryOut/DevEval"
DEFAULT_BASE_ENRE = "/data/data_public/riverbag/testRepoSummaryOut/DevEval"
DEFAULT_OUTPUT_CSV = "/data/data_public/riverbag/testRepoSummaryOut/DevEval/bm25_batch_metrics.csv"
DATA_JSONL = "/data/zxl/Search2026/DevEval/data.jsonl"


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> int:
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()


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


def _default_enre_path(project_key: str, base_enre: str) -> str:
    return os.path.join(base_enre, project_key, "report-enre.json")


def _project_key_from_project_root(project_root: str, source_code_dir: str, project_name: str) -> str:
    """
    从 project_root 中提取项目键（Source_Code 后到第一个 project_name，且包含该段）。

    例如：
    project_root = /data/lowcode_public/DevEval/Source_Code/System/mrjob/mrjob
    project_key = System/mrjob
    """
    norm_root = os.path.normpath(project_root)
    norm_source = os.path.normpath(source_code_dir)
    parts = [p for p in norm_root.split(os.sep) if p]
    source_parts = [p for p in norm_source.split(os.sep) if p]
    source_name = source_parts[-1] if source_parts else ""
    try:
        idx = parts.index(source_name)
    except ValueError:
        # 回退策略：直接取倒数三段中的中间两段
        if len(parts) >= 3:
            return "/".join(parts[-3:-1])
        if len(parts) >= 2:
            return "/".join(parts[-2:])
        return parts[-1]

    # 取 Source_Code 之后到第一个 project_name（包含该段）
    start = idx + 1
    end = len(parts)
    if project_name:
        name_norm = project_name.strip().lower()
        first_project_idx = -1
        for i in range(start, len(parts)):
            if parts[i].lower() == name_norm:
                first_project_idx = i
                break
        if first_project_idx >= start:
            end = first_project_idx + 1

    sub_parts = parts[start:end]
    if not sub_parts:
        # 兜底：至少返回最后两段中的可用部分
        if len(parts) >= 2:
            return "/".join(parts[-2:])
        return parts[-1] if parts else ""
    return "/".join(sub_parts)


def run_one_project(
    project_name: str,
    project_key: str,
    project_root: str,
    enre_json: str,
    base_search_out: str,
    source_code_dir: str,
    data_jsonl: str,
) -> Dict[str, Any]:
    project_dir = project_key

    project_base = os.path.join(base_search_out, project_key)
    methods_csv = os.path.join(project_base, "methods.csv")
    methods_desc_csv = os.path.join(project_base, "methods_with_desc.csv")
    filtered_path = os.path.join(project_base, "filtered.jsonl")
    refined_queries_cache_path = os.path.join(project_base, "refined_queries.json")

    if not os.path.exists(methods_csv):
        print(f"[skip] {project_name}: methods.csv not found at {methods_csv}")
        return {}
    if not os.path.exists(methods_desc_csv):
        print(f"[skip] {project_name}: methods_with_desc.csv not found at {methods_desc_csv}")
        return {}

    if not os.path.exists(filtered_path):
        print(f"[build] {project_name}: filtered.jsonl not found, building from {data_jsonl}")
        os.makedirs(project_base, exist_ok=True)
        with open(data_jsonl, "r") as infile, open(filtered_path, "w") as outfile:
            for line in infile:
                data = json.loads(line.strip())
                if data.get("project_path") == project_dir:
                    outfile.write(line)

    print(f"[run] {project_name} PROJECT_DIR={project_dir}")
    metrics = analyze_project(
        project_dir=project_dir,
        methods_csv=methods_csv,
        methods_desc_csv=methods_desc_csv,
        filtered_path=filtered_path,
        refined_queries_cache_path=refined_queries_cache_path,
        enre_json=enre_json,
        data_jsonl=data_jsonl,
    )
    metrics["project_name"] = project_name
    return metrics


def _append_rows_for_type(
    rows: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    project_name: str,
    metric_type: str,
) -> None:
    type_metrics = metrics.get(metric_type, {})
    num_examples = metrics.get("num_examples", 0)
    for k, v in type_metrics.items():
        rows.append(
            {
                "project_name": project_name,
                "project_dir": metrics.get("project_dir", ""),
                "metric_type": metric_type,
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
    p = argparse.ArgumentParser(description="Batch run BM25 search experiments from Excel project list")
    p.add_argument("--excel_path", default=EXCEL_PATH, help="Excel 路径，含项目名称、项目根目录，可选 ENRE路径")
    p.add_argument(
        "--base_search_out",
        default=DEFAULT_BASE_SEARCH_OUT,
        help="methods/methods_with_desc/filtered 等所在根目录",
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

    failed_log_path = os.path.join(args.base_search_out, "batch_bm25_failed_projects.log")
    os.makedirs(args.base_search_out, exist_ok=True)

    with open(failed_log_path, "w", encoding="utf-8") as log_file:
        tee_out = _Tee(sys.stdout, log_file)
        tee_err = _Tee(sys.stderr, log_file)
        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
            print(f"[log] Console output is also written to: {failed_log_path}")

            df = pd.read_excel(args.excel_path, sheet_name=args.sheet_name)
            df = _normalize_column_names(df)

            if "project_name" not in df.columns or "project_path" not in df.columns:
                raise SystemExit(
                    "Excel 需包含「项目名称」与「项目根目录」列（或 project_name / project_root）"
                )

            all_rows: List[Dict[str, Any]] = []
            per_project_rows: List[Dict[str, Any]] = []
            failed_log_lines: List[str] = []

            for _, row in df.iterrows():
                project_name = str(row["project_name"]).strip()
                project_root = str(row["project_path"]).strip()
                if not project_name or not project_root:
                    continue
                project_key = _project_key_from_project_root(
                    project_root=project_root,
                    source_code_dir=args.source_code_dir,
                    project_name=project_name,
                )
                enre_json = row.get("enre_json")
                if pd.isna(enre_json) or not str(enre_json).strip():
                    enre_json = _default_enre_path(project_key, args.base_enre)
                else:
                    enre_json = str(enre_json).strip()

                try:
                    metrics = run_one_project(
                        project_name=project_name,
                        project_key=project_key,
                        project_root=project_root,
                        enre_json=enre_json,
                        base_search_out=args.base_search_out,
                        source_code_dir=args.source_code_dir,
                        data_jsonl=args.data_jsonl,
                    )
                except Exception as e:
                    err_header = (
                        f"project_name={project_name} | project_key={project_key} | "
                        f"project_root={project_root} | error={type(e).__name__}: {e}"
                    )
                    tb = traceback.format_exc().strip()
                    print(f"[error] {err_header}")
                    failed_log_lines.append(err_header)
                    failed_log_lines.append(tb)
                    failed_log_lines.append("-" * 80)
                    continue

                if not metrics:
                    continue

                #_append_rows_for_type(per_project_rows, metrics, project_name, "signature")
                _append_rows_for_type(per_project_rows, metrics, project_name, "code")
                #_append_rows_for_type(per_project_rows, metrics, project_name, "desc")

            all_rows.extend(per_project_rows)

            # 计算总体指标：按项目的任务数加权
            if per_project_rows:
                import collections

                grouped: Dict[tuple, List[Dict[str, Any]]] = collections.defaultdict(list)
                for r in per_project_rows:
                    key = (r["metric_type"], int(r["k"]))
                    grouped[key].append(r)

                for (metric_type, k), rows in grouped.items():
                    total_tasks = sum(r["tasks"] for r in rows)
                    if total_tasks <= 0:
                        continue
                    P_total = sum(r["P"] * r["tasks"] for r in rows) / total_tasks
                    R_total = sum(r["R"] * r["tasks"] for r in rows) / total_tasks
                    F1_total = sum(r["F1"] * r["tasks"] for r in rows) / total_tasks
                    match_total = sum(r["match"] for r in rows)
                    pred_total = sum(r["pred"] for r in rows)
                    gt_total = sum(r["gt"] for r in rows)
                    all_rows.append(
                        {
                            "project_name": "ALL",
                            "project_dir": "",
                            "metric_type": metric_type,
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
                print(f"Saved batch BM25 metrics to {args.output_csv}")
            else:
                print("No metrics produced; nothing to save.")

            if failed_log_lines:
                print("[error-summary] Failed projects:")
                for line in failed_log_lines:
                    print(line)
                print(f"Saved failed project log to {failed_log_path}")


if __name__ == "__main__":
    main()

