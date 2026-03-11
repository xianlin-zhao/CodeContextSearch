"""
批量运行 graph_rag_gen：从 Excel 读取项目列表，按项目依次执行 load_enre_elements + generate_completions。

Excel 表格要求：
  - 至少两列：
    1) 项目名称（如 diffprivlib, mrjob, boto），列名可为「项目名称」或 project_name
    2) 项目根目录（即 GRAPH_PROJECT_PATH），列名可为「项目根目录」或 project_root
  - 可选列：enre_json 或「ENRE路径」——若不填则按约定路径推导：{base_enre}/{项目名称}/{项目名称}-report-enre.json
"""
import argparse
import os
import sys

# 保证可导入同目录模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from graph_rag_gen import generate_completions
from llm_clients import BackendName
from utils.task_recall import clear_enre_elements, load_enre_elements


EXCEL_PATH = "/data/zxl/Search2026/CodeContextSearch/src/generation/dev_eval/project_to_run/0311_5projects.xlsx"
SOURCE_CODE_DIR = "/data/lowcode_public/DevEval/Source_Code"
MODEL_NAME = "deepseek-v3"
MODEL_BACKEND_CHOICE = "openai"
# 默认路径前缀
DEFAULT_BASE_SEARCH_OUT = "/data/data_public/riverbag/testRepoSummaryOut/211"
DEFAULT_BASE_COMPLETION_OUT = "/data/zxl/Search2026/outputData/devEvalCompletionOut"
DEFAULT_BASE_ENRE = "/data/data_public/riverbag/testRepoSummaryOut/211"
SUBFOLDER = "0303_full"
GRAPH_SUBDIR = "graph_results_***all/PageRank-15-subgraph"


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """统一列名：项目名称 -> project_name, 项目根目录 -> project_root, ENRE路径 -> enre_json"""
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
    """按约定生成 ENRE 文件路径。若目录名与项目名不一致（如 diffpriv vs diffprivlib），需在 Excel 中显式填 enre_json。"""
    return os.path.join(base_enre, project_name, f"{project_name}-report-enre.json")


def run_one_project(
    project_name: str,
    project_root: str,
    enre_json: str,
    base_search_out: str,
    base_completion_out: str,
    source_code_dir: str,
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
    """对单个项目：清空并加载 ENRE，再调用 generate_completions。"""
    filtered_path = os.path.join(base_search_out, project_name, "filtered.jsonl")
    graph_dir_path = os.path.join(base_search_out, project_name, GRAPH_SUBDIR)
    output_jsonl = os.path.join(base_completion_out, project_name, SUBFOLDER, "graph_rag_completion.jsonl")
    print(f"filtered_path: {filtered_path}")
    print(f"graph_dir_path: {graph_dir_path}")
    print(f"output_jsonl: {output_jsonl}")

    if not os.path.exists(filtered_path):
        print(f"[skip] {project_name}: filtered.jsonl not found at {filtered_path}", file=sys.stderr)
        return
    if not os.path.exists(graph_dir_path):
        print(f"[skip] {project_name}: graph dir not found at {graph_dir_path}", file=sys.stderr)
        return

    clear_enre_elements()
    load_enre_elements(enre_json)

    debug_log = debug_log_override.strip() or None
    if not debug_log:
        debug_log = os.path.splitext(output_jsonl)[0] + "_debug.log"

    print(f"[run] {project_name} project_path={project_root}", file=sys.stderr)
    generate_completions(
        filtered_path=filtered_path,
        source_code_dir=source_code_dir,
        graph_dir_path=graph_dir_path,
        project_path=project_root,
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
    p = argparse.ArgumentParser(description="Batch run graph_rag_gen from Excel project list")
    p.add_argument("--excel_path", default=EXCEL_PATH, help="Excel 文件路径，含「项目名称」「项目根目录」列，可选「ENRE路径」")
    p.add_argument("--base_search_out", default=DEFAULT_BASE_SEARCH_OUT, help="Search 输出根目录")
    p.add_argument("--base_completion_out", default=DEFAULT_BASE_COMPLETION_OUT, help="Completion 输出根目录")
    p.add_argument("--base_enre", default=DEFAULT_BASE_ENRE, help="ENRE 根目录（未填 enre_json 时按 {base_enre}/{项目名}/{项目名}-report-enre.json 推导）")
    p.add_argument("--source_code_dir", default=SOURCE_CODE_DIR)
    p.add_argument("--backend", choices=["openai", "ollama", "mock"], default=MODEL_BACKEND_CHOICE)
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_tokens", type=int, default=0)
    p.add_argument("--timeout_s", type=float, default=120.0)
    p.add_argument("--max_tasks", type=int, default=0)
    p.add_argument("--sleep_s", type=float, default=0.0)
    p.add_argument("--sheet_name", default=0, help="Excel 工作表名或索引，默认第一个")
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
