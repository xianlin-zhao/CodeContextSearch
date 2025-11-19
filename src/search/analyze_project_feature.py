import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# 生成的feature路径
FEATURE_CSV = "/data/zxl/Search2026/outputData/repoSummaryOut/boto/1112/features.csv"
# DevEval数据集case的路径（json，不是数据集项目本身）
DATA_JSONL = "/data/lowcode_public/DevEval/data_have_dependency_cross_file.jsonl"


def analyze_project(project_path, output_path):
    # Create output directory under output_path using last two segments of project_path
    norm_path = os.path.normpath(project_path)
    parts = [p for p in norm_path.split(os.sep) if p]
    last_two = parts[-2:] if len(parts) >= 2 else parts
    dir_name = "_".join(last_two)
    output_dir = os.path.join(output_path, dir_name)
    print(f"Output directory: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Filter by project_path
    filtered_file = f"{output_dir}/filtered.jsonl"
    with open(DATA_JSONL, 'r') as infile, open(filtered_file, 'w') as outfile:
        for line in infile:
            data = json.loads(line.strip())
            if data.get('project_path') == '/'.join(last_two):
                outfile.write(line)
    
    # Step 2: Find similar clusters
    df = pd.read_csv(FEATURE_CSV)
    clusters = df.groupby('id')['desc'].first().reset_index()
    # 规范化 method_name：去掉参数，只保留第一个点后的部分，例如 mrjob.hadoop.main -> hadoop.main
    base_names = df['method_name'].astype(str).str.split('(').str[0]
    df['method_name_norm'] = base_names.str.split('.', n=1).str[1].fillna(base_names)
    method_names = df['method_name_norm'].unique().tolist()
    print(method_names)

    #model = SentenceTransformer('all-mpnet-base-v2')
    model = SentenceTransformer('all-MiniLM-L6-v2')
    cluster_embeddings = model.encode(clusters['desc'].tolist())
    
    with_clusters_file = f"{output_dir}/with_clusters.jsonl"
    with open(filtered_file, 'r') as f, open(with_clusters_file, 'w') as outfile:
        # 记录所有query top1的P，R，F1值， top3的P，R，F1值， top5的P，R，F1值
        top_1_match = 0
        top_3_match = 0
        top_5_match = 0
        top_1_pred = 0
        top_3_pred = 0
        top_5_pred = 0
        top_gt = 0

        for line in f:
            data = json.loads(line.strip())
            deps = []
            deps.extend(data['dependency']['intra_class'])
            deps.extend(data['dependency']['intra_file'])
            deps.extend(data['dependency']['cross_file'])
            deps = [dep for dep in deps if dep in method_names]

            top_gt += len(deps)

            query = data['requirement']['Functionality'] + ' ' + data['requirement']['Arguments']
            query_embedding = model.encode([query])
            similarities = cosine_similarity(query_embedding, cluster_embeddings)[0]
            top_1_indices = np.argsort(similarities)[-1:][::-1]

            top_3_indices = np.argsort(similarities)[-3:][::-1]

            top_5_indices = np.argsort(similarities)[-5:][::-1]

            # 计算top1：先从clusters获取id，再从原始df中获取method_name
            top_1_cluster_ids = clusters.iloc[top_1_indices]['id'].tolist()
            top_1_methods = []
            for cluster_id in top_1_cluster_ids:
                top_1_methods.extend(df[df['id'] == cluster_id]['method_name_norm'].tolist())
            top_1_pred += len(top_1_methods)
            for dep in deps:
                for method in top_1_methods:
                    if dep == method:
                        top_1_match += 1
                        break
            # 计算top3：先从clusters获取id，再从原始df中获取method_name
            top_3_cluster_ids = clusters.iloc[top_3_indices]['id'].tolist()
            top_3_methods = []
            for cluster_id in top_3_cluster_ids:
                top_3_methods.extend(df[df['id'] == cluster_id]['method_name_norm'].tolist())
            top_3_pred += len(top_3_methods)
            for dep in deps:
                for method in top_3_methods:
                    if dep == method:
                        top_3_match += 1
                        break
            # 计算top5：先从clusters获取id，再从原始df中获取method_name
            top_5_cluster_ids = clusters.iloc[top_5_indices]['id'].tolist()
            top_5_methods = []
            for cluster_id in top_5_cluster_ids:
                top_5_methods.extend(df[df['id'] == cluster_id]['method_name_norm'].tolist())
            top_5_pred += len(top_5_methods)
            for dep in deps:
                for method in top_5_methods:
                    if dep == method:
                        top_5_match += 1
                        break
            
            
        print(f"Top 1 Match: {top_1_match}")
        print(f"Top 1 Pred: {top_1_pred}")
        print(f"Top 1 P={(top_1_match/top_1_pred)*100:.2f}%")
        print(f"Top 1 R={(top_1_match/top_gt)*100:.2f}%")
        print(f"Top 1 F1={(2* (top_1_match/top_1_pred) * (top_1_match/top_gt) / ((top_1_match/top_1_pred) + (top_1_match/top_gt)))*100:.2f}%")
        print("--------------------------------")
        print(f"Top 3 Match: {top_3_match}")
        print(f"Top 3 Pred: {top_3_pred}")
        print(f"Top 3 P={(top_3_match/top_3_pred)*100:.2f}%")
        print(f"Top 3 R={(top_3_match/top_gt)*100:.2f}%")
        print(f"Top 3 F1={(2* (top_3_match/top_3_pred) * (top_3_match/top_gt) / ((top_3_match/top_3_pred) + (top_3_match/top_gt)))*100:.2f}%")
        print("--------------------------------")
        print(f"Top 5 Match: {top_5_match}")
        print(f"Top 5 Pred: {top_5_pred}")
        print(f"Top 5 P={(top_5_match/top_5_pred)*100:.2f}%")
        print(f"Top 5 R={(top_5_match/top_gt)*100:.2f}%")
        print(f"Top 5 F1={(2* (top_5_match/top_5_pred) * (top_5_match/top_gt) / ((top_5_match/top_5_pred) + (top_5_match/top_gt)))*100:.2f}%")
        print("--------------------------------")
    # # Step 3: Count dependency matches
    # results = []
    # with open(with_clusters_file, 'r') as f:
    #     total_gt_unfiltered = 0 # 真实依赖的个数
    #     total_gt = 0 # 真实依赖的个数
    #     total_pred = 0 # top1, top2, top3预测依赖的个数

    #     total_match = 0 # top1, top2, top3匹配的个数

    #     for line in f:
    #         data = json.loads(line.strip())
            
    #         deps = []
    #         deps.extend(data['dependency']['intra_class'])
    #         deps.extend(data['dependency']['intra_file'])
    #         deps.extend(data['dependency']['cross_file'])
    #         total_gt_unfiltered += len(deps)
    #         # 将deps映射到method_names中，如果method_names中存在，则保留，否则删除
    #         deps = [dep for dep in deps if dep in method_names]
    #         total_gt += len(deps)

    #         length_deps = len(deps)
    #         cluster_coverage = {}
            
    #         for cluster_id in data['top_3_clusters']:
    #             cluster_methods = df[df['id'] == cluster_id]['method_name'].tolist()
    #             # 这里的dep中函数的名字是：[mrjob.hadoop.HadoopJobRunner._hadoop_log_dirs, mrjob.hadoop.HadoopJobRunner.fs]
    #             # 这里的cluster_methods中函数的名字是：[mrjob.hadoop.HadoopJobRunner._hadoop_log_dirs(self, output_dir), mrjob.hadoop.HadoopJobRunner.fs(self)]
    #             match_count = 0
    #             total_pred += len(cluster_methods)
    #             for dep in deps:
    #                 for method in cluster_methods:
    #                     if dep in method:
    #                         match_count += 1
    #                         break
    #             total_match += match_count
    #             coverage_ratio = match_count / length_deps if length_deps > 0 else 0
    #             cluster_coverage[cluster_id] = coverage_ratio
            
    #         result = {
    #             'namespace': data['namespace'],
    #             'total_dependencies': length_deps,
    #             'top_3_clusters': data['top_3_clusters'],
    #             'cluster_coverage_ratio': cluster_coverage
    #         }
    #         results.append(result)
    # print(f"Total GT Unfiltered: {total_gt_unfiltered}")
    # print(f"Total GT: {total_gt}")
    # print(f"Total Pred: {total_pred}")
    # print(f"Total Match: {total_match}")
    # print(f"P={(total_match/total_pred)*100:.2f}%")
    # print(f"R={(total_match/total_gt)*100:.2f}%")
    # print(f"F1={(2* (total_match/total_pred) * (total_match/total_gt) / ((total_match/total_pred) + (total_match/total_gt)))*100:.2f}%")
    # # Save final results
    # analysis_file = f"{output_dir}/analysis.jsonl"
    # with open(analysis_file, 'w') as f:
    #     for result in results:
    #         f.write(json.dumps(result) + '\n')
    
    print(f"Analysis completed for {project_path}")
    print(f"Results saved in {output_dir}/ directory")


if __name__ == "__main__":
    # 项目源代码的路径
    project_path = "/data/lowcode_public/DevEval/Source_Code/Internet/boto"
    # 结果的保存路径（目前没用上）
    output_path = "/data/zxl/Search2026/outputData/devEvalSearchOut"
    analyze_project(project_path, output_path)
