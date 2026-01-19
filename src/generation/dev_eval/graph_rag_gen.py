import argparse
import os
import time
import sys
import json
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional, Tuple
import pandas as pd
import networkx as nx
import html

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
ENRE_JSON = "/data/data_public/riverbag/testRepoSummaryOut/Filited/mrjob/mrjob-report-enre.json"
GRAPH_DIR_PATH = "/data/zxl/Search2026/outputData/devEvalSearchOut/System_mrjob/0115/graph_results"
OUTPUT_COMPLETION_PATH = "/data/zxl/Search2026/outputData/devEvalCompletionOut/System_mrjob/0115/our_rag_completion.jsonl"

# 代码生成使用的大模型
MODEL_NAME = "deepseek-v3"
MODEL_BACKEND_CHOICE = "openai"

DEBUG = True  # 是否打印调试信息
GENERATION_FLAG = True  # 是否做代码生成，默认True，如果只是统计context recall，则设置为False

PROMPT_TEMPLATE = (
    "Please complete the function in the given Python code"
    "located at the end of the instuction based on relevant repository information.\n\n"
    "Constraints:\n"
    "- Output only the completion that should follow the given signature!\n"
    "- Do not repeat the signature!\n"
    "- Do not repeat the requirement comment!\n"
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

# 存储所有的变量，集合
variables_enre = set()

# 读入之前处理的所有method信息
def load_methods_info(methods_csv: str) -> None:
    df_methods = pd.read_csv(methods_csv)
    print("Loading METHODS_CSV...")
    
    for index, row in df_methods.iterrows():
        sig = str(row['method_signature'])
        method_sig_to_info[sig] = row.to_dict()

    print(f"Loaded {len(method_sig_to_info)} methods from CSV.")


def load_enre_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    variables = data.get('variables', [])
    
    for var in variables:
        if not var.get('category', '').startswith('Un'):
            # 把所有Variable类型的代码元素记录下来
            if var.get('category') == 'Variable':
                qname = var['qualifiedName']
                variables_enre.add(qname)


# 读入之前搜索的结果（包含是否match等指标）
def load_diagnostic_result(diagnostic_jsonl: str) -> list[Dict[str, Any]]:
    print("Loading diagnostic_jsonl...")
    diag_records = []
    
    with open(diagnostic_jsonl, 'r') as f:
        for line in f:
            if line.strip():
                diag_records.append(json.loads(line))
    
    return diag_records


def load_graph_result(task: DevEvalTask, graph_gml_path: str) -> list[Dict[str, Any]]:
    """
    Load a graph from a GML file and extract the 'sig', 'func_file', and 'method_code' attributes from each.
    """
    # 读取GML文件
    G = nx.read_gml(graph_gml_path)
    
    context_code_list = []

    # 遍历所有节点
    for node_id, attrs in G.nodes(data=True):
        # 提取 'sig', 'func_file', 'method_code' 属性
        sig = attrs.get("sig")
        func_file = attrs.get("func_file")
        method_code = attrs.get("method_code") # HTML 实体反转义（避免编码问题）
        if isinstance(sig, str):
            sig= html.unescape(sig)
        if isinstance(func_file, str):
            func_file = html.unescape(func_file)
        if isinstance(method_code, str):
            method_code = html.unescape(method_code)

        context_code = {
            'method_signature': sig,
            'func_file': func_file,
            'method_code': method_code,
        }
        context_code_list.append(context_code)
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


def _normalize_symbol(s: str) -> str:
    if "(" in s:
        return s.split("(", 1)[0]
    return s


# 计算搜索结果对于dependency的召回率
def compute_task_recall(
    dependency: Optional[list[str]],
    searched_context_code_list: list[Dict[str, Any]],
) -> Dict[str, Any]:
    dep = dependency or []
    dep_set = {x for x in dep}
    retrieved_set = {
        _normalize_symbol(str(x.get("method_signature", "")))
        for x in searched_context_code_list
        if isinstance(x, dict)
    }
    dep_total = len(dep_set)
    hit = len(dep_set & retrieved_set) if dep_total > 0 else 0
    
    for x in dep:
        if x in variables_enre:
            var_name = x.split('.')[-1]
            # 如果这个dependency里面的变量在某段搜到的代码中，就也认为是成功召回
            for context_code in searched_context_code_list:
                code_detail = context_code.get("method_code")
                if var_name in code_detail:
                    hit += 1
                    break

    recall = (hit / dep_total) if dep_total > 0 else None
    return {
        "dependency_total": dep_total,
        "dependency_hit": hit,
        "recall": recall,
    }


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

    processed = 0
    recall_sum = 0.0
    recall_count = 0
    recall_none_count = 0
    for record in iter_jsonl(filtered_path):
        task = parse_task(record)


        abs_file, signature = resolve_signature(
            source_code_dir, task.completion_path, task.signature_position
        )

        # 获取该任务对应的代码搜索结果（以图的形式）
        graph_gml_path = os.path.join(GRAPH_DIR_PATH, f"task_{processed + 1}_rank.gml")
        searched_context_code_list = load_graph_result(task, graph_gml_path)
        print(f"code len: {len(searched_context_code_list)}")
        context_code_in_prompt = assemble_context_code_into_prompt(searched_context_code_list)

        # 计算该任务的context recall
        recall_info = compute_task_recall(task.dependency, searched_context_code_list)
        if recall_info["recall"] is None:
            recall_none_count += 1
        else:
            recall_sum += float(recall_info["recall"])
            recall_count += 1

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
        
        if GENERATION_FLAG:
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

        if GENERATION_FLAG:
            write_jsonl_line(output_jsonl, {
                "namespace": task.namespace,
                "completion": completion,
                "idx": processed,
                "dependency": task.dependency,
                "recall": recall_info,
            })

        processed += 1
        if sleep_s > 0:
            time.sleep(sleep_s)
        if max_tasks is not None and processed >= max_tasks:
            break
    
    mean_recall = (recall_sum / recall_count) if recall_count > 0 else None
    print(
        json.dumps(
            {
                "recall_mean": mean_recall,
                "tasks_with_dependency": recall_count,
                "tasks_without_dependency": recall_none_count,
                "tasks_total": processed,
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
    )


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
    load_enre_json(ENRE_JSON)
    main()
