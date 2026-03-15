"""
从 jsonl 中只保留 dependency.cross_file 非空的记录，输出到新 jsonl。
"""
import argparse
import json
import sys


DEFAULT_INPUT_JSONL = "/data/zxl/Search2026/outputData/devEvalCompletionOut/0303_small_test/combined_filtered.jsonl"
DEFAULT_OUTPUT_JSONL = "/data/zxl/Search2026/outputData/devEvalCompletionOut/0303_small_cross_test/combined_filtered_cross.jsonl"

def main() -> None:
    p = argparse.ArgumentParser(description="Filter jsonl: keep only records with non-empty dependency.cross_file")
    p.add_argument("--input_jsonl", default=DEFAULT_INPUT_JSONL, help="输入 jsonl 路径")
    p.add_argument("--output", default=DEFAULT_OUTPUT_JSONL, help="输出 jsonl 路径")
    args = p.parse_args()

    kept = 0
    total = 0
    with open(args.input_jsonl, "r", encoding="utf-8") as f_in, open(
        args.output, "w", encoding="utf-8"
    ) as f_out:
        for line in f_in:
            line = line.rstrip("\n")
            if not line:
                continue
            total += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[warn] skip invalid json line: {e}", file=sys.stderr)
                continue
            dep = rec.get("dependency") or {}
            cross_file = dep.get("cross_file")
            if isinstance(cross_file, list) and len(cross_file) > 0:
                f_out.write(line + "\n")
                kept += 1

    print(f"Done: {kept}/{total} records kept -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
