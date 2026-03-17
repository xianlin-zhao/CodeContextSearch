import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"
from typing import Optional

import pandas as pd

from expand_and_rank_graph_exclude_TM_last import run_project


EXCEL_PATH = "/data/zxl/Search2026/CodeContextSearch/src/generation/dev_eval/project_to_run/0311_5projects.xlsx"
SOURCE_CODE_DIR = "/data/zxl/Search2026/Datasets/Source_Code"
DEFAULT_BASE_SEARCH_OUT = "/data/zxl/Search2026/outputData/devEvalSearchOut/0316_batch_workflow"
DEFAULT_BASE_ENRE = "/data/zxl/Search2026/outputData/devEvalSearchOut/0316_batch_workflow"
# Which embedding backend to use for personalization scores.
#   - "unixcoder": default, UniXcoder-based embeddings
#   - "bge-code": use BAAI/bge-code-v1 via sentence-transformers
EMBEDDING_BACKEND_KIND = "bge-code"


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
    从 project_root 中截取 PROJECT_PATH（Source_Code 后面的两段路径）。

    例如：
    project_root = /data/lowcode_public/DevEval/Source_Code/System/mrjob/mrjob
    PROJECT_PATH = System/mrjob
    """
    norm_root = os.path.normpath(project_root)
    norm_source = os.path.normpath(source_code_dir)
    parts = norm_root.split(os.sep)
    try:
        idx = parts.index(os.path.basename(norm_source))
    except ValueError:
        if len(parts) >= 3:
            return "/".join(parts[-3:-1])
        if len(parts) >= 2:
            return "/".join(parts[-2:])
        return parts[-1]

    start = idx + 1
    end = start + 2
    sub_parts = parts[start:end]
    return "/".join(sub_parts)


def run_one_project(
    *,
    project_name: str,
    project_root: str,
    enre_json: str,
    base_search_out: str,
    source_code_dir: str,
    embedding_backend_kind: str,
) -> None:
    project_base = os.path.join(base_search_out, project_name)
    methods_csv = os.path.join(project_base, "methods.csv")
    filtered_path = os.path.join(project_base, "filtered.jsonl")
    output_graph_path = os.path.join(project_base, "graph_results_***all")

    if not os.path.exists(methods_csv):
        print(f"[skip] {project_name}: methods.csv not found at {methods_csv}")
        return
    if not os.path.exists(filtered_path):
        print(f"[skip] {project_name}: filtered.jsonl not found at {filtered_path}")
        return
    if not os.path.exists(output_graph_path):
        print(f"[skip] {project_name}: graph dir not found at {output_graph_path}")
        return

    project_rel = _project_dir_from_project_root(project_root, source_code_dir)
    project_path = os.path.join(source_code_dir, project_rel)

    print(f"[run] {project_name} PROJECT_PATH={project_rel} (expand+rank)")
    run_project(
        methods_csv=methods_csv,
        enre_json=enre_json,
        filtered_path=filtered_path,
        output_graph_path=output_graph_path,
        project_path=project_path,
        embedding_backend_kind=embedding_backend_kind,
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Batch expand and rank graphs for multiple projects from Excel project list"
    )
    p.add_argument(
        "--excel_path",
        default=EXCEL_PATH,
        help="Excel 路径，含项目名称、项目根目录，可选 ENRE路径",
    )
    p.add_argument(
        "--base_search_out",
        default=DEFAULT_BASE_SEARCH_OUT,
        help="methods/filtered/graph_results_***all 等所在根目录",
    )
    p.add_argument(
        "--base_enre",
        default=DEFAULT_BASE_ENRE,
        help="ENRE 根目录（未填 enre_json 时推导用）",
    )
    p.add_argument(
        "--source_code_dir",
        default=SOURCE_CODE_DIR,
        help="DevEval 源代码根目录（用于从 project_root 截取 PROJECT_PATH）",
    )
    p.add_argument(
        "--embedding_backend_kind",
        default=EMBEDDING_BACKEND_KIND,
        choices=["unixcoder", "bge-code"],
    )
    p.add_argument("--sheet_name", default=0, help="Excel 工作表名或索引")
    args = p.parse_args()

    df = pd.read_excel(args.excel_path, sheet_name=args.sheet_name)
    df = _normalize_column_names(df)

    if "project_name" not in df.columns or "project_root" not in df.columns:
        raise SystemExit(
            "Excel 需包含「项目名称」与「项目根目录」列（或 project_name / project_root）"
        )

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

        run_one_project(
            project_name=project_name,
            project_root=project_root,
            enre_json=enre_json,
            base_search_out=args.base_search_out,
            source_code_dir=args.source_code_dir,
            embedding_backend_kind=args.embedding_backend_kind,
        )


if __name__ == "__main__":
    main()

