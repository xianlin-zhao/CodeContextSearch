import argparse
import time
import sys
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple
import pandas as pd

from llm_clients import BackendName, make_client
from utils.completion_postprocess import (
    extract_code_from_markdown,
    keep_only_completion,
    preview_text,
)
from utils.dev_eval_task import DevEvalTask, parse_task
from utils.jsonl_io import iter_jsonl, write_jsonl_line
from utils.source_code_utils import resolve_signature


SOURCE_CODE_DIR = "/data/lowcode_public/DevEval/Source_Code"
FILTERED_PATH = "/data/data_public/riverbag/testRepoSummaryOut/Filited/mrjob/filtered.jsonl"
METHODS_CSV = "/data/data_public/riverbag/testRepoSummaryOut/Filited/mrjob/methods.csv"
DIAGNOSTIC_JSONL = "/data/data_public/riverbag/testRepoSummaryOut/Filited/mrjob/diagnostic_bm25_code.jsonl"
OUTPUT_COMPLETION_PATH = "/data/zxl/Search2026/outputData/devEvalCompletionOut/System_mrjob/0104/bm25_rag/deepseek_completion.jsonl"
MODEL_NAME = "deepseek-v3"
MODEL_BACKEND_CHOICE = "openai"

# RAG使用的数据源，目前有几种："bm25", "unixcoder", "feature", "feature+bm25"
RAG_DATA_SOURCE = "bm25"

DEBUG = True  # 是否打印调试信息

PROMPT_TEMPLATE = (
    "Please complete the function in the given Python code"
    "located at the end of the instuction based on relevant repository information.\n\n"
    "Constraints:\n"
    "- Output only the completion that should follow the given signature!\n"
    "- Do not repeat the signature!\n"
    "- Do not repeat the requirement comment!\n\n"
    "- You can reference the code fragments from the repo to help you complete the function!\n\n"
    "Here are some relevant code fragments from the repo:\n"
    "{{context_code_in_prompt}}\n\n\n\n"
    "Input Code (You should complete):\n"
    "```Python\n"
    "{{signature}}\n\n"
    "{{requirement_comment}}\n\n"
    "```\n\n"
    "Completed Code:\n"
)


# 全局变量，存储所有method的信息，用于根据签名获取完整的代码
method_sig_to_info = {}

# 读入之前处理的所有method信息
def load_methods_info(methods_csv: str) -> None:
    df_methods = pd.read_csv(methods_csv)
    print("Loading METHODS_CSV...")
    
    for index, row in df_methods.iterrows():
        sig = str(row['method_signature'])
        method_sig_to_info[sig] = row.to_dict()

    print(f"Loaded {len(method_sig_to_info)} methods from CSV.")


# 读入之前搜索的结果（包含是否match等指标）
def load_diagnostic_result(diagnostic_jsonl: str) -> list[Dict[str, Any]]:
    print("Loading diagnostic_jsonl...")
    diag_records = []
    
    with open(diagnostic_jsonl, 'r') as f:
        for line in f:
            if line.strip():
                diag_records.append(json.loads(line))
    
    return diag_records


# 得到相应任务的搜索结果，作为之后给LLM的context
def get_searched_context_code(task: DevEvalTask, diag_record: Dict[str, Any]) -> list[Dict[str, Any]]:
    task_namespace = task.namespace
    preds = []
    try:
        if RAG_DATA_SOURCE == "bm25":
            preds = diag_record["bm25_code"]["top15"]["predictions"]
        elif RAG_DATA_SOURCE == "unixcoder":
            preds = diag_record["unixcoder_code"]["top15"]["predictions"]
        elif RAG_DATA_SOURCE == "feature":
            preds = diag_record["feature"]["top3"]["predictions"]
        elif RAG_DATA_SOURCE == "feature+bm25":
            # 因为后续还要在初始搜索结果的基础上进行扩展和筛选，因此先选取top10的
            preds = diag_record["hybrid"]["recall_top3_clusters"]["rank_top10"]["predictions"]
        
        # 过滤掉preds中method == task_namespace的项，也就是待补全的ground truth的代码
        filtered_preds = [p for p in preds if (p['method'] if '(' not in p['method'] else p['method'].split('(')[0]) != task_namespace]
        if len(filtered_preds) != len(preds):
            print(f"Attention! {len(preds) - len(filtered_preds)} items filtered out for task {task_namespace}")
        preds = filtered_preds
    except KeyError:
        print(f"Skipping task {task_namespace}: structure not matching data['feature']['top3']['predictions']")
        return []
    
    # 只要method的签名（完整版）
    pred_signatures = [p['method'] for p in preds]
    context_code_list = []
    for sig in pred_signatures:
        # 从method csv中找到签名对应的method具体信息
        if sig in method_sig_to_info:
            csv_info = method_sig_to_info[sig]
            context_code = {
                'method_signature': str(csv_info.get('method_signature', '')),
                'func_file': str(csv_info.get('func_file', '')),
                'method_code': str(csv_info.get('method_code', '')),
            }
            context_code_list.append(context_code)
        else:
            print(f"Warning! Signature {sig} not found in methods csv")
    
    return context_code_list


# 将搜索到的代码片段拼接起来，作为prompt中的context
def assemble_context_code_into_prompt(context_code_list: list[Dict[str, Any]]) -> str:
    context_code_in_prompt = ""
    for context_code in context_code_list:
        context_code_in_prompt += (
            f"{context_code['func_file']}\n"
            f"{context_code['method_code']}\n\n"
        )
    return context_code_in_prompt


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


def build_prompt(signature: str, requirement_comment: str, context_code_in_prompt: str) -> str:
    return (
        PROMPT_TEMPLATE.replace("{{signature}}", signature.rstrip("\n"))
        .replace("{{requirement_comment}}", requirement_comment)
        .replace("{{context_code_in_prompt}}", context_code_in_prompt)
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

    diag_records = load_diagnostic_result(DIAGNOSTIC_JSONL)

    processed = 0
    for record in iter_jsonl(filtered_path):
        task = parse_task(record)

        diag_record = diag_records[processed]

        abs_file, signature = resolve_signature(
            source_code_dir, task.completion_path, task.signature_position
        )

        # 获取该任务对应的代码搜索结果
        searched_context_code_list = get_searched_context_code(task, diag_record)
        context_code_in_prompt = assemble_context_code_into_prompt(searched_context_code_list)

        requirement_comment = format_requirement_as_comment(task.requirement_text)
        prompt = build_prompt(signature=signature, requirement_comment=requirement_comment,
                                context_code_in_prompt=context_code_in_prompt)

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
    load_methods_info(METHODS_CSV)
    main()
