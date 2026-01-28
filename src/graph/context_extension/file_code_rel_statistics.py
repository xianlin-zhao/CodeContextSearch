import argparse
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Tuple


PROJECT_DIR = "/data/lowcode_public/DevEval/Source_Code/Internet/boto"
ENRE_JSON = "/data/data_public/riverbag/testRepoSummaryOut/Filited/boto/boto-report-enre.json"
FILTERED_PATH = "/data/data_public/riverbag/testRepoSummaryOut/Filited/boto/filtered.jsonl"


@dataclass
class EnreIndex:
    nodes: Dict[str, Dict[str, Any]]
    adj: Dict[str, List[Tuple[str, Optional[str]]]]
    id_to_qname_norm: Dict[str, str]
    file_to_func_ids: Dict[str, List[str]]


def analyze_file_context_dependency(
    project_dir: str,
    file_path: str,
    method_to_complete: str,
    *,
    enre_json_path: str = ENRE_JSON,
    enre_index: Optional[EnreIndex] = None,
    top_k: int = 50,
    return_data: bool = False,
) -> Optional[Dict[str, Any]]:
    # 首先，要在enre report的variables里面找到所有那些File字段为file_path的、且类型为"Function"的代码元素
    # 他们正是这个文件里所有的函数，但是我们要特别去除method_to_complete函数（按照"qualifiedName"字段）
    # 对于剩下的每一个函数，我们都要做一个统计：
    # 找到这个函数在enre中的id，称为src_id
    # 找到enre中cells里面（也就代表每一条边，比如调用、使用），src为src_id的所有边
    # 这些边的终点dest，记录下所有这些dest的id，它们也就是在src这个函数中使用到的所有代码元素（可能是文件内部的、也可能是跨文件的）
    # method_to_complete是这个函数的全名，格式是比如mrjob.ssh.connect，我们要在dest中找到以它为前缀的所有代码元素
    # 这些代码元素也就是函数内部自己的元素，我们要记录下来，但之后并不关注这些元素
    # 我们真正关注的是其他的dest，我们要统计有哪些关系，比如src --CALL--> dest
    # 统计关系的时候，要区分终点的不同类型（比如USE class和USE function是不一样的）
    # 统计这个file_path里面除了method_to_complete函数之外，其他函数的上述逻辑，最后输出统计结果

    if enre_index is None:
        enre_index = load_enre_index(enre_json_path, project_dir)

    file_abs = _normalize_input_file_path(project_dir, file_path)
    method_prefix = method_to_complete

    func_ids = enre_index.file_to_func_ids.get(file_abs, [])
    analyzed_func_ids = [
        fid for fid in func_ids if enre_index.id_to_qname_norm.get(fid) != method_prefix
    ]

    relation_counter: Counter[Tuple[str, str]] = Counter()


    for src_id in analyzed_func_ids:
        internal_dest_ids: set[str] = set()
        external_dest_ids: set[str] = set()
        for dest_id, kind in enre_index.adj.get(src_id, []):
            dest_node = enre_index.nodes.get(dest_id)
            if dest_node is None:
                continue
            dest_qname_norm = enre_index.id_to_qname_norm.get(dest_id, "")
            if dest_qname_norm.startswith(method_prefix):
                internal_dest_ids.add(dest_id)
                continue

            external_dest_ids.add(dest_id)
            edge_kind = kind or "UNKNOWN"
            dest_category = dest_node.get("category") or "Unknown"

            relation_counter[(edge_kind, dest_category)] += 1

    def _sort_counter(counter: Counter, limit: int) -> List[Tuple[Any, int]]:
        return sorted(counter.items(), key=lambda x: (-x[1], str(x[0])))[:limit]

    result: Dict[str, Any] = {
        "file_path": file_abs,
        "method_to_complete": method_to_complete,
        "functions_in_file": len(func_ids),
        "functions_analyzed": len(analyzed_func_ids),
        "top_relations": [
            {"kind": k, "dest_category": c, "count": cnt}
            for (k, c), cnt in _sort_counter(relation_counter, top_k)
        ],
    }

    if return_data:
        return result
    print(json.dumps(result, ensure_ascii=False))
    return None


def _keep_enre_node(var: dict) -> bool:
    if not var.get('category', ''):
        return False
    cat = var.get('category', '')
    if cat.startswith("Unknown") or cat.startswith("Ambiguous"):
        return False
    if cat == 'Unresolved Attribute':
        # 必须存在"File"字段，且File必须包含"/"，否则可能是外部库文件里面的属性
        if ('File' not in var) or ('/' not in var.get('File', '')):
            return False
    return True

def _normalize_input_file_path(project_dir: str, file_path: str) -> str:
    file_path = file_path.strip()
    if os.path.isabs(file_path):
        return os.path.normpath(file_path)
    return os.path.normpath(os.path.join(project_dir, file_path))


def _normalize_enre_file_path(project_dir: str, enre_file_path: Optional[str]) -> Optional[str]:
    if not enre_file_path:
        return None
    if os.path.isabs(enre_file_path):
        return os.path.normpath(enre_file_path)
    return os.path.normpath(os.path.join(project_dir, enre_file_path))


@lru_cache(maxsize=4)
def load_enre_index(enre_json_path: str, project_dir: str) -> EnreIndex:
    with open(enre_json_path, "r") as f:
        data = json.load(f)

    variables = data.get("variables", [])
    cells = data.get("cells", [])

    nodes: Dict[str, Dict[str, Any]] = {}
    id_to_qname_norm: Dict[str, str] = {}
    file_to_func_ids: DefaultDict[str, List[str]] = defaultdict(list)

    for var in variables:
        if not _keep_enre_node(var):
            continue
        vid = str(var.get("id"))
        qname = var.get("qualifiedName") or ""
        category = var.get("category") or ""
        file_abs = _normalize_enre_file_path(project_dir, var.get("File"))

        nodes[vid] = {"qualifiedName": qname, "category": category}
        id_to_qname_norm[vid] = qname

        if category == "Function" and file_abs is not None:
            file_to_func_ids[file_abs].append(vid)

    adj: DefaultDict[str, List[Tuple[str, Optional[str]]]] = defaultdict(list)
    seen_by_src: DefaultDict[str, set[Tuple[str, Optional[str]]]] = defaultdict(set)

    for cell in cells:
        src = cell.get("src")
        dest = cell.get("dest")
        if src is None or dest is None:
            continue
        src = str(src)
        dest = str(dest)
        if src not in nodes or dest not in nodes:
            continue
        kind = (cell.get("values") or {}).get("kind")
        pair = (dest, kind)
        if pair in seen_by_src[src]:
            continue
        seen_by_src[src].add(pair)
        adj[src].append(pair)

    return EnreIndex(
        nodes=nodes,
        adj=dict(adj),
        id_to_qname_norm=id_to_qname_norm,
        file_to_func_ids=dict(file_to_func_ids),
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", default=PROJECT_DIR)
    parser.add_argument("--enre-json", default=ENRE_JSON)
    parser.add_argument("--top-k", type=int, default=100)

    subparsers = parser.add_subparsers(dest="cmd", required=True)

    one = subparsers.add_parser("one")
    one.add_argument("--file", required=True)
    one.add_argument("--method", required=True)

    batch = subparsers.add_parser("filtered")
    batch.add_argument("--filtered-jsonl", default=FILTERED_PATH)

    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        if args.cmd == "one":
            analyze_file_context_dependency(
                args.project_dir,
                args.file,
                args.method,
                enre_json_path=args.enre_json,
                top_k=args.top_k,
                return_data=False,
            )
            return 0

        if args.cmd == "filtered":
            enre_index = load_enre_index(args.enre_json, args.project_dir)
            with open(args.filtered_jsonl, "r") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    task = json.loads(line)
                    method_to_complete = task.get("namespace") or ""
                    completion_path = task.get("completion_path") or ""
                    project_path = task.get("project_path") or ""
                    if project_path and completion_path.startswith(project_path):
                        completion_path = completion_path[len(project_path) :].lstrip("/\\")
                    example_id = task.get("example_id", i + 1)

                    res = analyze_file_context_dependency(
                        args.project_dir,
                        completion_path,
                        method_to_complete,
                        enre_json_path=args.enre_json,
                        enre_index=enre_index,
                        top_k=args.top_k,
                        return_data=True,
                    )
                    if res is None:
                        continue
                    res["example_id"] = example_id
                    print(json.dumps(res, ensure_ascii=False))
            return 0
    except BrokenPipeError:
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
