import os
from typing import Tuple


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
def resolve_signature(
    source_code_dir: str, completion_path: str, signature_pos: Tuple[int, int]
) -> Tuple[str, str]:
    abs_file = os.path.join(source_code_dir, completion_path)
    signature = read_line_range(abs_file, signature_pos[0], signature_pos[1])
    return abs_file, signature

