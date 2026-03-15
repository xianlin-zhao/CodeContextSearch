"""
从 Excel 读取 project_name 列表，将各项目下指定子目录中的 jsonl 文件合并为一个大的 jsonl。

Excel 需包含 project_name 列（或「项目名称」）；路径规则：
  {BASE_COMPLETION_OUT}/{project_name}/{SUBFOLDER}/{JSONL_FILENAME}
"""
import argparse
import os
import sys

import pandas as pd

# excel路径，里面的project_name列是项目名称，用于拼接路径
DEFAULT_EXCEL = "/data/zxl/Search2026/CodeContextSearch/src/generation/dev_eval/project_to_run/0311_5projects.xlsx"
# base目录，里面会包含各个项目的文件夹，项目文件夹下会有各自的filtered.jsonl文件
DEFAULT_BASE_FILTERED = "/data/data_public/riverbag/testRepoSummaryOut/211"
# 可能的子目录，比如项目文件夹下嵌套着某个文件夹，filtered.jsonl在这个文件夹里(这里为空表示直接在项目文件夹下)
DEFAULT_SUBFOLDER = ""
# 要合并哪个文件(统一的名字)
DEFAULT_JSONL_FILENAME = "filtered.jsonl"
# 合并后的jsonl路径
OUTPUT_COMBINED_JSONL = "/data/zxl/Search2026/outputData/devEvalCompletionOut/0303_small_test/combined_filtered.jsonl"


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """统一列名：项目名称 -> project_name"""
    col_map = {}
    for c in df.columns:
        c_str = str(c).strip()
        if c_str in ("项目名称", "project_name"):
            col_map[c] = "project_name"
    return df.rename(columns=col_map)


def main() -> None:
    p = argparse.ArgumentParser(description="Merge per-project jsonl files into one, using project list from Excel")
    p.add_argument("--excel_path", default=DEFAULT_EXCEL, help="Excel 路径，需含 project_name 列")
    p.add_argument("--base_completion_out", default=DEFAULT_BASE_FILTERED, help="Completion 输出根目录")
    p.add_argument("--subfolder", default=DEFAULT_SUBFOLDER, help="每个项目下的子目录，如 0303_full")
    p.add_argument("--jsonl_name", default=DEFAULT_JSONL_FILENAME, help="要合并的 jsonl 文件名，如 no_context_completion.jsonl")
    p.add_argument("--output", "-o", default=OUTPUT_COMBINED_JSONL, help="合并后的输出 jsonl 路径")
    p.add_argument("--sheet", type=int, default=0, help="Excel 工作表索引，默认 0")
    args = p.parse_args()

    if not os.path.isfile(args.excel_path):
        print(f"Error: Excel not found: {args.excel_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_excel(args.excel_path, sheet_name=args.sheet)
    df = _normalize_column_names(df)
    if "project_name" not in df.columns:
        print("Error: Excel 需包含「项目名称」或 project_name 列", file=sys.stderr)
        sys.exit(1)

    project_names = df["project_name"].astype(str).str.strip()
    project_names = project_names[project_names.str.len() > 0].tolist()

    total_lines = 0
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w", encoding="utf-8") as out_f:
        for name in project_names:
            path = os.path.join(args.base_completion_out, name, args.subfolder, args.jsonl_name)
            if not os.path.isfile(path):
                print(f"[skip] {name}: not found {path}", file=sys.stderr)
                continue
            count = 0
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    out_f.write(line + "\n")
                    count += 1
            total_lines += count
            print(f"[ok] {name}: {count} lines -> {path}", file=sys.stderr)

    print(f"Done: {total_lines} lines written to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
