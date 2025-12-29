# CodeContextSearch

## 数据集

DevEval位于服务器 `/data/lowcode_public/DevEval`，其中data_have_dependency_cross_file.jsonl是在所有task中包含跨文件API调用的task



## 整体流程

目前的流程是这样的：

1.使用RepoSummary方法，对于一个repo生成feature summary；

2.对于DevEval中这个repo的每一个sample，搜索出结果（有多种搜索方式，优先采用feature+BM25的方式），都是method；

3.对于初步搜索所得的结果，分析这些method之间的关系，建图-**初始子图**；

4.基于一些规则，对初始子图的某些节点进行**扩展**（具体的规则仍在探索中）；

5.对扩展后的子图的结果进行筛选，保留重要程度较高的节点（目前尝试使用一些图算法，不基于LLM筛选），作为最终的**推理子图**



目前这些步骤是分离的，并没有实现成完整的接口，主要是为了探索每个阶段的方法。

目前还没有实现将推理子图的内容作为context，生成代码，近期需要实现，以跑通完整的流程。

目前的实验都仅在DevEval的个别项目上开展，还没有在完整的benchmark上进行实验。之后除了DevEval，还会在SWE-Bench上针对代码定位任务进行实验。



### 1. RepoSummary

在``/src/summarization/`目录下，创建一个`.env`文件：

```
OPENAI_API_KEY=xxxx
OPENAI_BASE_URL=xxxx
```

在`/src/summarization/main.py`中，配置main中的`project_root`和`output_dir`，再配置`STRATEGY`和`IS_PARALLEL`

在聚类时，分为文件聚类和函数聚类，有很多聚类参数可以调整。建议在给一个repo生成所有的feature summary之前，先看看聚类结果（可以通过设置`GENERATE_DESCRIPTION`来决定是否要生成feature summary，还是只看聚类结果而不生成summary），如果聚类的数量不符合预期，建议先调整参数改进一下效果，然后再生成feature description（毕竟调大模型API是要花钱的hhh）

跑完整个RepoSummary的过程后会在output_dir下生成一些csv文件：methods.csv, methods_with_desc.csv, file_adj_matrix.csv, method_adj_matrix.csv, features.csv。我们主要关心features.csv和methods.csv，前者包含所有生成的feature，后者包含repo中所有的method信息。



### 2. 在DevEval上进行method-level代码搜索

这一步的目的是根据query找到repo中可能相关的代码。

传统的方法通常用BM25, code embedding等方式，这里探索使用RepoSummary聚类出的feature来提高搜索效果。当然，这一步的提升幅度仍有限制，所以会有后面的建图、扩展、推理等步骤，以期获得更为匹配的搜索结果。

`/src/search`目录下，包含多种搜索方式：

* 使用生成的feature进行搜索，将匹配到的feature关联到的所有method作为答案  `/src/search/feature_and_bm25_search.py`
* 将method签名按蛇形命名法分割后，用query根据BM25搜索最相关的method  `/src/search/feature_and_bm25_search.py`
* 将method的具体代码切词后，用query根据BM25搜索最相关的method  `/src/search/feature_and_bm25_search.py`  （这3种检索方式都在一个python文件里）
* 使用code embedding进行搜索，UnixCoder这个模型可以为自然语言生成embedding，也为代码生成embedding，两者映射到同一向量空间  `/src/search/unixcoder_based_search.py`
* 先使用feature搜索，然后根据BM25（将代码切词后构建BM25索引）对搜索结果进行重排，相当于先通过全局的功能特征进行搜索，再通过局部的词语进行匹配，目前来看**这种搜索方式效果最好，推荐使用**！  `/src/search/feature_BM25Code.py`
* 先使用feature搜索，然后根据代码描述信息（CodeT5生成的method-level summary，非常简要）对搜索结果进行重排  `/src/search/feature_BM25Desc.py`
* 先使用feature搜索，然后根据代码的embedding（UnixCoder）进行重排  ``/src/search/feature_UnixCoder.py``



运行搜索的代码时，修改：

* PROJECT_PATH：要搜索的repo，以类似System/mrjob的格式
* FEATURE_CSV：RepoSummary生成的features
* METHODS_CSV：RepoSummary提取出的repo中所有method的信息
* FILTERED_PATH：这里可以是新建的文件，用于保存特定repo相关的任务
* DATA_JSONL：DevEval数据集所有任务的那个jsonl文件
* NEED_METHOD_NAME_NORM：举例来说，DevEval中代码元素的名称是类似mrjob.xx.yy.zz，但我们RepoSummary如果是作用于完整的repo，就会生成类似mrjob.mrjob.xx.yy.zz，这时候需要将这个变量设置为True，从而进行规范化。如果我们RepoSummary只作用于repo内部mrjob这个文件夹，那会生成mrjob.xx.yy.zz，此时就不需要规范化，设置为False即可
* USE_REFINED_QUERY：可以设置为False。这是为了测试LLM改写搜索词的效果，可以不用



运行后，会在FILTERED_PATH这个jsonl所在的目录下，生成搜索结果的文件，如：`diagnostic_hybrid.jsonl`

如果想研究json的具体内容，可复制到 https://www.json.cn/ 更清晰地查看。



### 3. 初始子图的构建

在`/src/graph/`目录下，`build_graph.py`可以根据上一步的搜索结果，对这些搜到的method建图——初始子图。

运行建图代码时，修改：

* METHODS_CSV：RepoSummary提取出的repo中所有method的信息
* ENRE_JSON：RepoSummary生成的中间结果，是用xjtu-enre工具对repo做静态分析得到的结果，文件名称通常是xxxx-report-enre.json
* FILTERED_PATH：这里可以是新建的文件，用于保存特定repo相关的任务
* DIAGNOSTIC_JSONL：上一步搜索结果的文件
* OUTPUT_GRAPH_PATH：输出结果保存在哪个文件夹。对于repo的每一个任务，都会生成一个`task_{task_id}_ori.gml`文件，这是networkx库使用的标准格式文件，保存图的节点和边

另外，DIAGNOSTIC_JSONL的内容较为复杂，在读入搜索结果时，目前使用的是`preds = rec["hybrid"]["recall_top3_clusters"]["rank_top10"]["predictions"]`的结果来建图，也就是在top3的feature所关联到的method中，根据BM25重排得到的top10 method，可根据需要来修改。



目前这里只分析了搜到的method之间的关系，之后考虑增加file和directory实体，这样可以表示更多关系，希望提升后续扩展和筛选的效果。



### 4. 子图扩展及筛选（推理）（探索中）

在`/src/graph/`目录下，`expand_and_rank_graph.py`可以基于初始子图，从某些节点沿某些关系进行适当的扩展，目的是找到更多可能相关的context；在扩展之后，为了防止后续输入LLM的context过多，保证只输入重要的context，还会对节点的重要性进行评估，目前使用的pagerank算法。

运行扩展及筛选的代码时，修改：

* METHODS_CSV：RepoSummary提取出的repo中所有method的信息
* ENRE_JSON：RepoSummary生成的中间结果，是用xjtu-enre工具对repo做静态分析得到的结果，文件名称通常是xxxx-report-enre.json
* FILTERED_PATH：这里可以是新建的文件，用于保存特定repo相关的任务
* OUTPUT_GRAPH_PATH：输出结果保存在哪个文件夹，可以和建图的结果保存在一个文件夹里，对于repo的每一个任务，先会生成一个`task_{task_id}_mid.gml`文件，代表扩展之后的结果，然后会生成`task_{task_id}_rank.gml`，代表pagerank筛选之后，排名top 15的节点所对应的子图



可运行`visualize_gml.py`，这是为了调试方便，写的一个可视化界面（trae写的），通过设置BASE_DIR（保存图的文件夹），可以展示这个文件夹下的所有图。



TODO：需要确认一下pagerank的效果，后续可根据目前调研到的论文，尝试更多方法（以下是一些KGQA的论文）

* HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models (Neurips 24)
* From RAG to Memory: Non-Parametric Continual Learning for Large Language Models (ICML 25)
  * 根据query抽取子图，Personalized Pagerank，最后按节点分数选top-n
* G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering (Neurips 24)
  * Prize-Collecting Steiner Tree 找相关性最大、规模可控的子图
* AGRAG: Advanced Graph-based Retrieval-Augmented Generation for LLMs (arxiv 2511)
  * MCMI 最小代价最大影响子图生成



### 在Dev上进行代码生成

TODO

目前的计划是：用以上步骤得到的搜索结果，作为context提供给LLM。同时，这些context不只是简单的文本拼接，而是以图的形式，不仅包含具体的代码，还包括这些代码之间的关系（这个idea在ICSE 26的一篇论文GRACE中也提到了）。

期望的结果是：由于前面的搜索结果更加准确相关，能搜到待完成的method所依赖的更多代码元素，所以LLM的代码生成效果有所提高，且使用了更多应该用的依赖，即Pass@k和Recall@k会提高。







