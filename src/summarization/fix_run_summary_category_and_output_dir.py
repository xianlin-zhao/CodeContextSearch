import argparse
import csv
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


def infer_category_from_project_root(project_root: str) -> str:
    path = Path(project_root)
    parts = path.parts
    if "Source_Code" in parts:
        idx = parts.index("Source_Code")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return ""


def move_dir_safely(src: Path, dst: Path) -> Tuple[bool, str]:
    if src == dst:
        return False, "same_path"
    if not src.exists():
        return False, "src_missing"

    dst.parent.mkdir(parents=True, exist_ok=True)

    if not dst.exists():
        shutil.move(str(src), str(dst))
        return True, "moved"

    # Merge mode when destination already exists.
    moved_any = False
    for item in src.iterdir():
        target_item = dst / item.name
        if target_item.exists():
            # Keep existing destination item; skip conflict.
            continue
        shutil.move(str(item), str(target_item))
        moved_any = True

    # Remove source directory if it is empty after merge attempts.
    try:
        src.rmdir()
    except OSError:
        pass

    return moved_any, "merged_or_skipped"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix run_summary.csv category/output_dir and move misplaced output dirs.")
    parser.add_argument(
        "--csv",
        default="/data/data_public/riverbag/testRepoSummaryOut/DevEval/run_summary.csv",
        help="Path to run_summary.csv",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows: List[Dict[str, str]] = list(reader)

    if not fieldnames:
        raise ValueError(f"CSV has no header: {csv_path}")

    fixed_category_count = 0
    fixed_output_dir_count = 0
    moved_dirs = 0
    move_status_counts: Dict[str, int] = {}

    for row in rows:
        project_root = (row.get("project_root") or "").strip()
        old_category = (row.get("category") or "").strip()
        project = (row.get("project") or "").strip()
        old_output_dir = (row.get("output_dir") or "").strip()

        inferred_category = infer_category_from_project_root(project_root)
        if inferred_category and inferred_category != old_category:
            row["category"] = inferred_category
            fixed_category_count += 1
        else:
            inferred_category = old_category

        if not old_output_dir or not inferred_category or not project:
            continue

        old_out = Path(old_output_dir)
        dev_eval_root = old_out.parent.parent if old_out.parent.parent.name else old_out.parent.parent
        new_out = dev_eval_root / inferred_category / project

        if str(new_out) != old_output_dir:
            row["output_dir"] = str(new_out)
            fixed_output_dir_count += 1

            changed, status = move_dir_safely(old_out, new_out)
            if changed:
                moved_dirs += 1
            move_status_counts[status] = move_status_counts.get(status, 0) + 1

    rows.sort(key=lambda r: ((r.get("category") or ""), (r.get("project") or "")))

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"csv={csv_path}")
    print(f"rows_total={len(rows)}")
    print(f"fixed_category_count={fixed_category_count}")
    print(f"fixed_output_dir_count={fixed_output_dir_count}")
    print(f"moved_dirs={moved_dirs}")
    print(f"move_status_counts={move_status_counts}")


if __name__ == "__main__":
    main()
