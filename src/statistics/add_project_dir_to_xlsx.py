import argparse
import os
from typing import Optional

import pandas as pd


def compute_project_dir(project_name: str, project_path: str) -> str:
    name = str(project_name).strip()
    path = str(project_path).strip()
    if not name or not path:
        return ""

    path = path.replace("\\", "/")
    parts = [p for p in path.split("/") if p]
    if not parts:
        return ""

    name_norm = name.lower()
    idx: Optional[int] = None
    for i, part in enumerate(parts):
        if part.lower() == name_norm:
            idx = i
            break

    if idx is None or idx == 0:
        return ""

    return f"{parts[idx - 1]}/{parts[idx]}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/data/data_public/riverbag/CodeContextSearch/docs/deveval_project_path.xlsx",
    )
    parser.add_argument(
        "--output",
        default="/data/data_public/riverbag/CodeContextSearch/docs/deveval_project_path_with_dir.xlsx",
    )
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    df = pd.read_excel(input_path, dtype=str).fillna("")
    for col in ("project_name", "project_path"):
        if col not in df.columns:
            raise ValueError(f"missing column: {col}; columns={list(df.columns)}")

    project_dirs = [
        compute_project_dir(project_name, project_path)
        for project_name, project_path in zip(
            df["project_name"].astype(str).tolist(),
            df["project_path"].astype(str).tolist(),
        )
    ]

    if "project_dir" in df.columns:
        df["project_dir"] = project_dirs
    else:
        insert_at = len(df.columns)
        if "project_path" in df.columns:
            insert_at = int(list(df.columns).index("project_path")) + 1
        df.insert(insert_at, "project_dir", project_dirs)

    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_excel(output_path, index=False)
    print(output_path)


if __name__ == "__main__":
    main()
