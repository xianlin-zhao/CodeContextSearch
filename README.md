# CodeContextSearch

## 数据集

DevEval位于服务器 `/data/lowcode_public/DevEval`，其中data_have_dependency_cross_file.jsonl是在所有task中包含跨文件API调用的task

## 运行方法

目前分为两部分：首先对于一个repo生成feature summary，然后在devEval上评价API搜索的效果。

目前还不涉及后续的建图、推理、代码生成，而且现在只在一个数据集上进行初步实验，后续可能会使用LocBench、SweBench等数据集，来完成多种下游任务。

### RepoSummary

在``/src/summarization/`目录下，创建一个`.env`文件：

```
OPENAI_API_KEY=xxxx
OPENAI_BASE_URL=xxxx
```



在`/src/summarization/main.py`中，配置main中的project_root和output_dir，再配置STRATEGY和IS_PARALLEL

在聚类时，分为文件聚类和函数聚类，有很多聚类参数可以调整。建议在给一个repo生成所有的feature summary之前，先看看聚类结果（可以把后面生成feature description的部分注释掉），如果聚类的数量不符合预期，建议先调整参数改进一下效果，然后再生成feature description（毕竟调大模型API是要花钱的hhh）

跑完整个RepoSummary的过程后会在output_dir下生成一些csv文件：methods.csv, methods_with_desc.csv, file_adj_matrix.csv, method_adj_matrix.csv, features.csv。我们主要关心features.csv和methods.csv，前者包含所有生成的feature，后者包含repo中所有的method信息。

### 在devEval上评价API搜索

`/src/search/feature_and_bm25_search.py`（记得修改main中的project_path）包含三种搜索方式：

* 使用生成的feature进行搜索，将匹配到的feature关联到的所有method作为答案
* 将method签名按蛇形命名法分割后，用query根据BM25搜索最相关的method
* 将method的具体代码切词后，用query根据BM25搜索最相关的method

`/src/search/unixcoder_based_search.py`使用code embedding进行搜索，unixcoder这个模型可以为自然语言生成embedding，也为代码生成embedding，两者映射到同一向量空间

运行这两个文件之前，要修改每个文件开头的几个路径（其中FILTERED_PATH可以自己指定一个，它的含义是从data_have_dependency_cross_file.jsonl中读出特定repo的task）





