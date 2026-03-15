import pandas as pd


# KEEP_PROJECT_NAMES = {"boto", "alembic", "diffprivlib", "mopidy"}
KEEP_PROJECT_NAMES = {"mingus"}


def main() -> None:
    input_path = "/data/data_public/riverbag/CodeContextSearch/docs/deveval_project_path_with_dir.xlsx"
    output_path = "/data/data_public/riverbag/CodeContextSearch/docs/deveval_four_projects.xlsx"

    df = pd.read_excel(input_path, dtype=str).fillna("")
    if "project_name" not in df.columns:
        raise ValueError(f"missing column: project_name; columns={list(df.columns)}")

    project_name_norm = df["project_name"].astype(str).str.strip().str.lower()
    out_df = df.loc[project_name_norm.isin(KEEP_PROJECT_NAMES)].copy()

    out_df.to_excel(output_path, index=False)
    print(output_path)


if __name__ == "__main__":
    main()
