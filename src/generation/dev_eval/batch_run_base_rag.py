"""
批量运行 base_rag_gen：从 Excel 读取项目列表，按项目依次执行 load_methods_info + load_enre_elements + generate_completions。

Excel 表格要求（与 batch_run_graph_rag 相同格式可复用）：
  - 至少两列：项目名称（project_name / 项目名称）、项目根目录（project_root / 项目根目录）
  - 可选列：enre_json / ENRE路径，不填则按 {base_enre}/{项目名}/{项目名}-report-enre.json 推导
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from base_rag_gen import generate_completions
from llm_clients import BackendName
from utils.task_recall import clear_enre_elements, load_enre_elements


EXCEL_PATH = "/data/zxl/Search2026/CodeContextSearch/src/generation/dev_eval/project_to_run/0311_5projects.xlsx"
SOURCE_CODE_DIR = "/data/lowcode_public/DevEval/Source_Code"
MODEL_NAME = "deepseek-v3"
MODEL_BACKEND_CHOICE = "openai"
# RAG使用的数据源，目前有几种："bm25", "unixcoder", "feature", "feature+bm25"
RAG_DATA_SOURCE = "bm25"
# 诊断结果文件名，与 RAG 数据源对应（如 unixcoder -> diagnostic_unixcoder_code.jsonl）
DEFAULT_DIAGNOSTIC_FILENAME = "diagnostic_bm25_code.jsonl"
COMPLETION_FILENAME = "bm25_rag_completion.jsonl"  # 一定要与RAG数据源对应！

DEFAULT_BASE_SEARCH_OUT = "/data/data_public/riverbag/testRepoSummaryOut/211"
DEFAULT_BASE_COMPLETION_OUT = "/data/zxl/Search2026/outputData/devEvalCompletionOut"
DEFAULT_BASE_ENRE = "/data/data_public/riverbag/testRepoSummaryOut/211"
SUBFOLDER = "0303_full"


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


def run_one_project(
    project_name: str,
    project_root: str,
    enre_json: str,
    base_search_out: str,
    base_completion_out: str,
    source_code_dir: str,
    diagnostic_filename: str,
    rag_data_source: str,
    backend: BackendName,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int | None,
    timeout_s: float,
    max_tasks: int | None,
    sleep_s: float,
    debug_log_override: str = "",
) -> None:
    """对单个项目：清空 ENRE，再调用 generate_completions（内部会 load_methods_info + load_enre_elements）。"""
    filtered_path = os.path.join(base_search_out, project_name, "filtered.jsonl")
    methods_csv = os.path.join(base_search_out, project_name, "methods.csv")
    diagnostic_jsonl = os.path.join(base_search_out, project_name, diagnostic_filename)
    output_jsonl = os.path.join(base_completion_out, project_name, SUBFOLDER, COMPLETION_FILENAME)

    if not os.path.exists(filtered_path):
        print(f"[skip] {project_name}: filtered.jsonl not found at {filtered_path}", file=sys.stderr)
        return
    if not os.path.exists(methods_csv):
        print(f"[skip] {project_name}: methods.csv not found at {methods_csv}", file=sys.stderr)
        return
    if not os.path.exists(diagnostic_jsonl):
        print(f"[skip] {project_name}: diagnostic not found at {diagnostic_jsonl}", file=sys.stderr)
        return

    clear_enre_elements()
    # generate_completions 内部会 load_methods_info(methods_csv) 和 load_enre_elements(enre_json)
    debug_log = debug_log_override.strip() or None
    if not debug_log:
        debug_log = os.path.splitext(output_jsonl)[0] + "_debug.log"

    print(f"[run] {project_name} rag_data_source={rag_data_source}", file=sys.stderr)
    generate_completions(
        filtered_path=filtered_path,
        source_code_dir=source_code_dir,
        methods_csv=methods_csv,
        enre_json=enre_json,
        diagnostic_jsonl=diagnostic_jsonl,
        rag_data_source=rag_data_source,
        output_jsonl=output_jsonl,
        debug_log_path_override=debug_log,
        backend=backend,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        timeout_s=timeout_s,
        max_tasks=max_tasks,
        sleep_s=sleep_s,
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Batch run base_rag_gen from Excel project list")
    p.add_argument("--excel_path", default=EXCEL_PATH, help="Excel 路径，含项目名称、项目根目录，可选 ENRE路径")
    p.add_argument("--base_search_out", default=DEFAULT_BASE_SEARCH_OUT, help="filtered/methods/diagnostic 所在根目录")
    p.add_argument("--base_completion_out", default=DEFAULT_BASE_COMPLETION_OUT, help="Completion 输出根目录")
    p.add_argument("--base_enre", default=DEFAULT_BASE_ENRE, help="ENRE 根目录（未填 enre_json 时推导用）")
    p.add_argument("--diagnostic_filename", default=DEFAULT_DIAGNOSTIC_FILENAME,
                   help="诊断 jsonl 文件名，如 diagnostic_unixcoder_code.jsonl")
    p.add_argument("--rag_data_source", default=RAG_DATA_SOURCE,
                   choices=["bm25", "unixcoder", "feature", "feature+bm25"])
    p.add_argument("--source_code_dir", default=SOURCE_CODE_DIR)
    p.add_argument("--backend", choices=["openai", "ollama", "mock"], default=MODEL_BACKEND_CHOICE)
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_tokens", type=int, default=0)
    p.add_argument("--timeout_s", type=float, default=120.0)
    p.add_argument("--max_tasks", type=int, default=0)
    p.add_argument("--sleep_s", type=float, default=0.0)
    p.add_argument("--sheet_name", default=0, help="Excel 工作表名或索引")
    args = p.parse_args()

    max_tokens = args.max_tokens if args.max_tokens and args.max_tokens > 0 else None
    max_tasks = args.max_tasks if args.max_tasks and args.max_tasks > 0 else None

    df = pd.read_excel(args.excel_path, sheet_name=args.sheet_name)
    df = _normalize_column_names(df)

    if "project_name" not in df.columns or "project_root" not in df.columns:
        print("Error: Excel 需包含「项目名称」与「项目根目录」列（或 project_name / project_root）", file=sys.stderr)
        sys.exit(1)

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
            base_completion_out=args.base_completion_out,
            source_code_dir=args.source_code_dir,
            diagnostic_filename=args.diagnostic_filename,
            rag_data_source=args.rag_data_source,
            backend=args.backend,
            model=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=max_tokens,
            timeout_s=args.timeout_s,
            max_tasks=max_tasks,
            sleep_s=args.sleep_s,
        )


if __name__ == "__main__":
    main()
