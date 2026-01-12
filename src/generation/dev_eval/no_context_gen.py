import argparse
import json
import os
import time
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple

from llm_clients import BackendName, make_client


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


@dataclass(frozen=True)
class DevEvalTask:
    namespace: str
    completion_path: str
    signature_position: Tuple[int, int]
    requirement_text: str
    indent: Optional[int] = None


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid jsonl at line {line_num}: {path}") from e


def build_requirement_text(requirement: Any) -> str:
    functionality = requirement.get("Functionality")
    arguments = requirement.get("Arguments")

    return functionality + "\n" + arguments


# 把jsonl里的一条记录解析成我们规定的task结构
def parse_task(record: Dict[str, Any]) -> DevEvalTask:
    namespace = record.get("namespace")
    completion_path = record.get("completion_path")
    signature_position = record.get("signature_position")

    if not isinstance(namespace, str) or not namespace:
        raise ValueError("Task missing valid 'namespace'")
    if not isinstance(completion_path, str) or not completion_path:
        raise ValueError(f"Task {namespace} missing valid 'completion_path'")
    if (
        not isinstance(signature_position, (list, tuple))
        or len(signature_position) != 2
        or not all(isinstance(x, int) for x in signature_position)
    ):
        raise ValueError(f"Task {namespace} missing valid 'signature_position'")

    requirement_text = build_requirement_text(record.get("requirement"))
    indent = record.get("indent")
    indent = indent if isinstance(indent, int) and indent >= 0 else None

    start, end = int(signature_position[0]), int(signature_position[1])
    if start <= 0 or end <= 0 or end < start:
        raise ValueError(
            f"Task {namespace} has invalid signature_position: {signature_position}"
        )

    return DevEvalTask(
        namespace=namespace,
        completion_path=completion_path,
        signature_position=(start, end),
        requirement_text=requirement_text,
        indent=indent,
    )


def read_line_range(file_path: str, start_line: int, end_line: int) -> str:
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    if start_line < 1 or end_line < 1:
        raise ValueError("start_line/end_line must be 1-based positive integers")
    if end_line > len(lines):
        raise ValueError(
            f"Requested lines {start_line}-{end_line} exceed file length {len(lines)}: {file_path}"
        )
    return "".join(lines[start_line - 1 : end_line])


# 从原始的py文件里，根据函数签名所在的行号来获得函数签名
def resolve_signature(source_code_dir: str, completion_path: str, signature_pos: Tuple[int, int]) -> Tuple[str, str]:
    abs_file = os.path.join(source_code_dir, completion_path)
    signature = read_line_range(abs_file, signature_pos[0], signature_pos[1])
    return abs_file, signature


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


# 从大模型返回的结果中提取出代码
def extract_code_from_markdown(text: str) -> str:
    if not text:
        return ""
    pattern = re.compile(r"```(?:[a-zA-Z0-9_+-]+)?\s*\n([\s\S]*?)\n?```", re.MULTILINE)
    matches = pattern.findall(text)
    if matches:
        candidate = max(matches, key=lambda s: len(s.strip()))
        return candidate.strip("\n")
    return text.strip("\n")


# 打印内容，方便调试
def preview_text(text: str, max_chars: int = 1200) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... (truncated, total_chars={len(text)})"

def remove_triple_quoted_block_containing_anchor(text: str, anchor: str) -> str:
    if not text or not anchor:
        return text
    out = text
    while True:
        anchor_idx = out.find(anchor)
        if anchor_idx == -1:
            break
        start_sq = out.rfind("'''", 0, anchor_idx)
        start_dq = out.rfind('"""', 0, anchor_idx)
        start = max(start_sq, start_dq)
        if start == -1:
            break
        delimiter = out[start : start + 3]
        end = out.find(delimiter, anchor_idx)
        if end == -1:
            break
        out = out[:start] + out[end + 3 :]
    return out


# 从大模型返回的结果中，提取出补全的代码，去掉可能的函数签名、多行注释
def keep_only_completion(
    completion_text: str,
    *,
    signature: str,
    requirement_comment: str,
    requirement_text: str,
) -> str:
    if not completion_text:
        return ""

    text = completion_text.replace("\r\n", "\n").replace("\r", "\n")

    # 如果补全结果里直接就包含需求注释，删掉
    req_comment_norm = (requirement_comment or "").replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    if req_comment_norm:
        text = text.replace(req_comment_norm, "")

    # 删除可能的多行注释
    req_text_norm = (requirement_text or "").strip()
    if req_text_norm:
        anchor = req_text_norm[:30]
        text = remove_triple_quoted_block_containing_anchor(text, anchor)

    # 删除可能的函数签名
    sig_norm = (signature or "").replace("\r\n", "\n").replace("\r", "\n").strip("\n")
    if sig_norm:
        sig_idx = text.rfind(sig_norm)
        if sig_idx != -1:
            text = text[sig_idx + len(sig_norm) :]

    return text.lstrip("\n")


def build_prompt(signature: str, requirement_comment: str) -> str:
    return (
        PROMPT_TEMPLATE.replace("{{signature}}", signature.rstrip("\n"))
        .replace("{{requirement_comment}}", requirement_comment)
    )


# 把一条记录写入jsonl文件，采用添加模式，不覆盖已有内容
def write_jsonl_line(path: str, record: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


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
