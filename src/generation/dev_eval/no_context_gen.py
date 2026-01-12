import argparse
import time
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple

from llm_clients import BackendName, make_client
from utils.completion_postprocess import (
    extract_code_from_markdown,
    keep_only_completion,
    preview_text,
)
from utils.dev_eval_task import parse_task
from utils.jsonl_io import iter_jsonl, write_jsonl_line
from utils.source_code_utils import resolve_signature


SOURCE_CODE_DIR = "/data/lowcode_public/DevEval/Source_Code"
FILTERED_PATH = "/data/data_public/riverbag/testRepoSummaryOut/mrjob/1:3/filtered.jsonl"
OUTPUT_COMPLETION_PATH = "/data/zxl/Search2026/outputData/devEvalCompletionOut/System_mrjob/0104/no_context/deepseek_completion.jsonl"
MODEL_NAME = "deepseek-v3"
MODEL_BACKEND_CHOICE = "openai"

DEBUG = True  # 是否打印调试信息

PROMPT_TEMPLATE = (
    "Please complete the function in the given Python code.\n\n"
    "Constraints:\n"
    "- Output only the completion that should follow the given signature!\n"
    "- Do not repeat the signature!\n"
    "- Do not repeat the requirement comment!\n\n"
    "Input Code:\n"
    "```Python\n"
    "{{signature}}\n\n"
    "{{requirement_comment}}\n\n"
    "```\n\n"
    "Completed Code:\n"
)


# 将需求文本格式化为多行注释，用于拼接在函数签名下面
def format_requirement_as_comment(requirement_text: str) -> str:
    if not requirement_text:
        return ""
    lines = requirement_text.splitlines()
    indent_str = "    "  # 每个缩进4个空格
    delimiter = '"""'  # 多行注释用"""表示
    escaped_delimiter = '\\"\\"\\"'
    lines = [line.replace(delimiter, escaped_delimiter) for line in lines]

    content = "\n".join(indent_str + line if line else indent_str for line in lines)
    return f"{indent_str}{delimiter}\n{content}\n{indent_str}{delimiter}\n"


def build_prompt(signature: str, requirement_comment: str) -> str:
    return (
        PROMPT_TEMPLATE.replace("{{signature}}", signature.rstrip("\n"))
        .replace("{{requirement_comment}}", requirement_comment)
    )


def generate_completions(
    *,
    filtered_path: str,
    source_code_dir: str,
    output_jsonl: str,
    backend: BackendName,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: Optional[int],
    timeout_s: float,
    max_tasks: Optional[int],
    sleep_s: float,
) -> None:

    # 清空输出结果的jsonl文件
    with open(output_jsonl, "w", encoding="utf-8"):
        pass

    client = make_client(
        backend=backend,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        timeout_s=timeout_s,
    )

    processed = 0
    for record in iter_jsonl(filtered_path):
        task = parse_task(record)

        abs_file, signature = resolve_signature(
            source_code_dir, task.completion_path, task.signature_position
        )
        requirement_comment = format_requirement_as_comment(task.requirement_text)
        prompt = build_prompt(signature=signature, requirement_comment=requirement_comment)

        if DEBUG:
            print(f"[debug] namespace={task.namespace}", file=sys.stderr)
            print(f"[debug] file={abs_file}", file=sys.stderr)
            print(f"[debug] signature_position={task.signature_position}", file=sys.stderr)
            print("[debug] signature:\n" + preview_text(signature), file=sys.stderr)
            print("[debug] requirement_comment:\n" + preview_text(requirement_comment), file=sys.stderr)
            print("[debug] prompt:\n" + preview_text(prompt), file=sys.stderr)

        raw_completion = client.generate(prompt)
        extracted_completion = extract_code_from_markdown(raw_completion)
        completion = keep_only_completion(
            extracted_completion,
            signature=signature,
            requirement_comment=requirement_comment,
            requirement_text=task.requirement_text,
        )
        
        if DEBUG:
            print("[debug] raw_completion:\n" + preview_text(raw_completion), file=sys.stderr)
            print("[debug] extracted_completion:\n" + preview_text(extracted_completion), file=sys.stderr)
            print("[debug] final_completion:\n" + preview_text(completion), file=sys.stderr)

        write_jsonl_line(output_jsonl, {
            "namespace": task.namespace,
            "completion": completion,
            "idx": processed
        })

        processed += 1
        if sleep_s > 0:
            time.sleep(sleep_s)
        if max_tasks is not None and processed >= max_tasks:
            break


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--filtered_path", default=FILTERED_PATH)
    p.add_argument("--source_code_dir", default=SOURCE_CODE_DIR)
    p.add_argument("--output", default=OUTPUT_COMPLETION_PATH)
    p.add_argument("--backend", choices=["openai", "ollama", "mock"], default=MODEL_BACKEND_CHOICE)
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--temperature", type=float, default=0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_tokens", type=int, default=0)
    p.add_argument("--timeout_s", type=float, default=120.0)
    p.add_argument("--max_tasks", type=int, default=0)
    p.add_argument("--sleep_s", type=float, default=0.0)
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    max_tokens = args.max_tokens if args.max_tokens and args.max_tokens > 0 else None
    max_tasks = args.max_tasks if args.max_tasks and args.max_tasks > 0 else None
    generate_completions(
        filtered_path=args.filtered_path,
        source_code_dir=args.source_code_dir,
        output_jsonl=args.output,
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
