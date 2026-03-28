# 代码上下文图构建与分析

本目录包含基于搜索结果和 ENRE 静态分析报告构建、分析、扩展及可视化代码上下文图（Code Context Graph）的脚本。

## 文件概览

1.  **`build_graph.py`** (构建初始图)
    *   **功能**: 为每个任务构建初始的代码依赖图 (`_ori.gml`)。它将搜索结果（来自 `diagnostic_feature.jsonl`）与静态分析数据（来自 `enre.json`）结合，将搜索到的预测结果映射为实际的代码实体及其相互关系。
    *   **输入**: `methods.csv`, `mrjob-report-enre.json`, `diagnostic_feature.jsonl`.
    *   **输出**: 输出目录下的 `task_{id}_ori.gml` 文件。

2.  **`analyze_ori_graph_diff.py`** (差异分析)
    *   **功能**: 诊断工具。将生成的初始图与真值（Ground Truth, GT）依赖进行对比。它会报告哪些 GT 方法未被包含在图中，并尝试在 ENRE 报告中找到它们，以解释缺失原因。
    *   **输入**: `methods.csv`, `mrjob-report-enre.json`, `filtered.jsonl` (包含 GT), 以及生成的 `.gml` 文件。

3.  **`expand_and_rank_graph.py`** (扩展与排序)
    *   **功能**: 增强初始图。
        *   **Expand (扩展)**: 根据静态分析数据，添加相关联的邻居节点（调用者/被调用者）以丰富上下文信息。
        *   **Rank (排序)**: 使用 `UniXcoder` 嵌入向量对节点进行评分和排序，可能用于过滤掉相关性较低的节点。
    *   **输入**: `_ori.gml` 文件。
    *   **输出**: `_mid.gml` (扩展后) 和 `_rank.gml` (排序/过滤后) 文件。

4.  **`batch_build_graph.py`** (批量构建图)
    *   **功能**: 根据 Excel 项目列表，批量为多个项目运行 `build_graph.py`。
    *   **参数说明**:
        *   `--excel_path`: 包含项目信息的 Excel 文件路径（需包含 `project_name` 列）。
        *   `--base_search_out`: 搜索结果（`methods.csv`, `filtered.jsonl` 等）存放的根目录。
        *   `--base_enre`: ENRE 报告存放的根目录。
        *   `--output_csv`: 批量构建结果的汇总 CSV 输出路径。
    *   **示例**:
        ```bash
        python batch_build_graph.py --excel_path projects.xlsx --base_search_out ./outputs --base_enre ./enre_data --output_csv build_summary.csv
        ```

5.  **`batch_expand_and_rank_graph.py`** (批量扩展与排序)
    *   **功能**: 批量为多个项目运行扩展与排序逻辑。
    *   **参数说明**:
        *   `--excel_path`: 包含项目信息的 Excel 文件路径（需包含 `project_name` 和 `project_root` 列）。
        *   `--base_search_out`: 包含构建好的图（`graph_results_***all` 目录）的根目录。
        *   `--source_code_dir`: DevEval 源代码的根目录。
        *   `--embedding_backend_kind`: 嵌入向量后端，可选 `unixcoder` 或 `bge-code`。
    *   **示例**:
        ```bash
        python batch_expand_and_rank_graph.py --excel_path projects.xlsx --base_search_out ./outputs --source_code_dir ./src --embedding_backend_kind bge-code
        ```

6.  **`batch_compare_graph_recall.py`** (批量评估召回率)
    *   **功能**: 批量计算多个项目的图召回率指标（Ori vs Mid vs Rank），并汇总到 CSV。
    *   **参数说明**:
        *   `--excel_path`: 包含项目信息的 Excel 文件路径。
        *   `--base_search_out`: 包含各项目图处理结果的根目录。
        *   `--output_csv`: 汇总指标的 CSV 输出路径。
    *   **示例**:
        ```bash
        python batch_compare_graph_recall.py --excel_path projects.xlsx --base_search_out ./outputs --output_csv recall_summary.csv
        ```

7.  **`visualize_gml.py`** (可视化)
    *   **功能**: 一个简单的 Flask Web 应用程序，使用 `pyvis` 交互式地可视化生成的 `.gml` 文件。
    *   **用法**: 运行脚本后，在浏览器中访问 `http://localhost:5000`。

## 运行顺序

1.  **配置路径**: 在运行任何脚本之前，请打开文件并确认文件路径变量（如 `METHODS_CSV`, `ENRE_JSON`, `OUTPUT_GRAPH_PATH`）与您的实际数据位置一致。

2.  **构建初始图**:
    ```bash
    python build_graph.py
    ```

3.  **(可选) 分析覆盖率**:
    ```bash
    python analyze_ori_graph_diff.py
    ```

4.  **(可选) 扩展和排序**:
    ```bash
    python expand_and_rank_graph.py
    ```

5.  **可视化**:
    ```bash
    python visualize_gml.py
    ```

## 批量运行指南

批量处理脚本通常需要一个 Excel 配置文件，指定项目列表及其根目录。

1.  **批量构建图**:
    ```bash
    python batch_build_graph.py \
      --excel_path "/path/to/projects.xlsx" \
      --base_search_out "/path/to/search_results" \
      --base_enre "/path/to/enre_reports" \
      --output_csv "batch_build_metrics.csv"
    ```

2.  **批量扩展与排序**:
    ```bash
    python batch_expand_and_rank_graph.py \
      --excel_path "/path/to/projects.xlsx" \
      --base_search_out "/path/to/search_results" \
      --source_code_dir "/path/to/source_code" \
      --embedding_backend_kind "bge-code"
    ```

3.  **批量评估指标**:
    ```bash
    python batch_compare_graph_recall.py \
      --excel_path "/path/to/projects.xlsx" \
      --base_search_out "/path/to/search_results" \
      --output_csv "batch_recall_summary.csv"
    ```
