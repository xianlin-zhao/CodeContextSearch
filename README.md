# CodeContextSearch

## 数据集

DevEval位于服务器 `/data/lowcode_public/DevEval`，其中data_have_dependency_cross_file.jsonl是在所有task中包含跨文件API调用的task



## 整体流程

目前的流程是这样的：

1.使用RepoSummary方法，对于一个repo生成feature summary；

2.对于DevEval中这个repo的每一个sample，搜索出结果（有多种搜索方式，优先采用feature top3的方式），都是method；

3.对于初步搜索所得的结果，分析这些method之间的关系，建图-**初始图**；

4.基于一些规则，对初始子图的某些节点进行扩展，得到**扩展图**；

5.对扩展图的结果进行筛选，保留重要程度较高的节点（节点基础得分+加分，personalized pagerank），作为最终的**推理子图**；

6.将推理子图的内容作为context，组织prompt，LLM根据自然语言需求和搜索到的context，生成代码



目前这些步骤是分离的，并没有实现成完整的接口，主要是为了探索每个阶段的方法。

目前的实验都仅在DevEval的个别项目上开展，还没有在完整的benchmark上进行实验。之后除了DevEval，还会再找其他的benchmark (EvoCodeBench)进行实验。



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

* 使用生成的feature进行搜索，会把target method对应的feature提高到top1。可以指定选择top-n的feature作为搜索结果（**毕设最终采用的方案**）  `/src/search/feature_based_search.py`

* 使用生成的feature进行搜索，将匹配到的feature关联到的所有method作为答案  `/src/search/feature_and_bm25_search.py`
* 将method签名按蛇形命名法分割后，用query根据BM25搜索最相关的method  `/src/search/feature_and_bm25_search.py`
* 将method的具体代码切词后，用query根据BM25搜索最相关的method  `/src/search/feature_and_bm25_search.py`  （这3种检索方式都在一个python文件里）
* 使用code embedding进行搜索，UnixCoder这个模型可以为自然语言生成embedding，也为代码生成embedding，两者映射到同一向量空间  `/src/search/unixcoder_based_search.py`
* 先使用feature搜索，然后根据BM25（将代码切词后构建BM25索引）对搜索结果进行重排，相当于先通过全局的功能特征进行搜索，再通过局部的词语进行匹配，目前来看**这种搜索方式效果很好**！  `/src/search/feature_BM25Code.py`
* 先使用feature搜索，然后根据代码描述信息（CodeT5生成的method-level summary，非常简要）对搜索结果进行重排  `/src/search/feature_BM25Desc.py`
* 先使用feature搜索，然后根据代码的embedding（UnixCoder）进行重排  ``/src/search/feature_UnixCoder.py``



运行搜索的代码时，修改：

* PROJECT_PATH：要搜索的repo，以类似System/mrjob的格式，用于过滤出这个repo对应的任务
* FEATURE_CSV：RepoSummary生成的features
* METHODS_CSV：RepoSummary提取出的repo中所有method的信息
* ENRE_JSON：enre工具生成的report，json格式
* FILTERED_PATH：这里可以是新建的文件，用于保存特定repo相关的任务
* DATA_JSONL：DevEval数据集所有任务的那个jsonl文件
* NEED_METHOD_NAME_NORM：举例来说，DevEval中代码元素的名称是类似mrjob.xx.yy.zz，但我们RepoSummary如果是作用于完整的repo，就会生成类似mrjob.mrjob.xx.yy.zz，这时候需要将这个变量设置为True，从而进行规范化。如果我们RepoSummary只作用于repo内部mrjob这个文件夹，那会生成mrjob.xx.yy.zz，此时就不需要规范化，设置为False即可
* USE_REFINED_QUERY：可以设置为False。这是为了测试LLM改写搜索词的效果，可以不用



运行后，会在FILTERED_PATH这个jsonl所在的目录下，生成搜索结果的文件，如：`diagnostic_hybrid.jsonl`

如果想研究json的具体内容，可复制到 https://www.json.cn/ 更清晰地查看。



### 3. 初始图的构建

在`/src/graph/`目录下，`build_graph.py`可以根据上一步的搜索结果，对这些搜到的method建图——初始图。

运行建图代码时，修改：

* METHODS_CSV：RepoSummary提取出的repo中所有method的信息
* ENRE_JSON：RepoSummary生成的中间结果，是用xjtu-enre工具对repo做静态分析得到的结果，文件名称通常是xxxx-report-enre.json
* FILTERED_PATH：这里可以是新建的文件，用于保存特定repo相关的任务
* DIAGNOSTIC_JSONL：上一步搜索结果的文件
* OUTPUT_GRAPH_PATH：输出结果保存在哪个文件夹。对于repo的每一个任务，都会生成一个`task_{task_id}_ori.gml`文件，这是networkx库使用的标准格式文件，保存图的节点和边
* REMOVE_FIRST_DOT_PREFIX：与上面的NEED_METHOD_NAME_NORM类似，现在我们生成的代码元素名称格式与DevEval相同，所以这里默认为False
* PREFIX：如果REMOVE_FIRST_DOT_PREFIX为True，则要注意修改这里的项目名称

另外，DIAGNOSTIC_JSONL的内容较为复杂，在读入搜索结果时，目前使用的是`preds = rec["feature"]["top3"]["predictions"]`的结果来建图（这是根据实验，得到的比较好的结果），也就是选择top3的features关联到的所有methods。



### 4. 子图扩展及筛选（推理）

在`/src/graph/`目录下，`expand_and_rank_graph.py`可以基于初始子图，从某些节点沿某些关系进行适当的扩展，目的是找到更多可能相关的context；在扩展之后，为了防止后续输入LLM的context过多，保证只输入重要的context，还会对节点的重要性进行评估，目前使用的personalized pagerank算法。

运行扩展及筛选的代码时，修改：

* METHODS_CSV：RepoSummary提取出的repo中所有method的信息
* ENRE_JSON：RepoSummary生成的中间结果，是用xjtu-enre工具对repo做静态分析得到的结果，文件名称通常是xxxx-report-enre.json
* FILTERED_PATH：这里可以是新建的文件，用于保存特定repo相关的任务
* OUTPUT_GRAPH_PATH：输出结果保存在哪个文件夹，可以和建图的结果保存在一个文件夹里，对于repo的每一个任务，先会生成一个`task_{task_id}_mid.gml`文件，代表扩展之后的结果，然后会生成`task_{task_id}_rank.gml`，代表pagerank筛选之后，排名top 15的节点所对应的子图
* PROJECT_PATH：项目源代码路径的**上一层**，比如说项目源代码在`/data/lowcode_public/DevEval/Source_Code/Internet/boto`，这里就应该填`/data/lowcode_public/DevEval/Source_Code/Internet`，这个参数主要是用于跟代码元素所在文件的路径做拼接
* TOP_KS：最终选择top-k分数的节点作为推理子图，保存起来，保存到子文件夹PageRank-{k}-subgraph
* EMBEDDING_BACKEND_KIND：计算节点分数时，会计算query与节点代码的相似度，选择使用哪种模型进行embedding计算，可以选择"bge-code"或"unixcoder"，前者的效果更好，但速度会略慢



完成扩展和筛选之后，可运行`compare_graph_recall.py`，能够比较初始搜索后、扩展后、筛选后的召回率，效果提高/降低的任务数量，用于评估各阶段贡献。会生成一个csv文件和log文件。

可运行`visualize_gml.py`，这是为了调试方便，写的一个可视化界面，通过设置BASE_DIR（保存图的文件夹），可以展示这个文件夹下的所有图。





相关的KGQA论文

* HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models (Neurips 24)
* From RAG to Memory: Non-Parametric Continual Learning for Large Language Models (ICML 25)
  * 根据query抽取子图，Personalized Pagerank，最后按节点分数选top-n
* G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering (Neurips 24)
  * Prize-Collecting Steiner Tree 找相关性最大、规模可控的子图
* AGRAG: Advanced Graph-based Retrieval-Augmented Generation for LLMs (arxiv 2511)
  * MCMI 最小代价最大影响子图生成

 

### 5. 在DevEval上进行代码生成

这一步要做的是：用以上步骤得到的搜索结果，作为context提供给LLM，做代码生成。

这些context不只是简单的文本拼接，而是以图的形式，不仅包含具体的代码，还包括这些代码之间的关系（此idea在ICSE 26的一篇论文GRACE中也提到了）。

期望的结果是：由于前面的搜索结果更加准确相关，能搜到待完成的method所依赖的更多代码元素，所以LLM的代码生成效果有所提高，且使用了更多应该用的依赖，即Pass@k和Recall@k会提高。

在`/src/generation`目录下，首先需要建一个`.env`文件，里面是用于代码生成的API key信息，可以复用RepoSummary阶段的`.env`文件。

目前有3种代码生成的方式：

* No Context：直接给LLM自然语言需求和待生成函数的签名，没有任何别的内容 `src/generation/no-contex_gen.py`
* BM25 / embedding based-RAG：用自然语言需求在repo的函数里进行检索，选top-N最相似的，作为context。可以用BM25 / Unixcoder / feature / feature + BM25 rerank这些检索方式 `src/generation/base_rag_gen.py`，关于不同的检索方式究竟用什么搜索结果，可以看`get_searched_context_code`函数
* Graph based-RAG：用之前构建的推理子图，用图上的内容作为context `src/generation/graph_rag_gen.py`

生成代码时，修改：

* SOURCE_CODE_DIR：DevEval数据集源代码的总路径，比如/data/lowcode_public/DevEval/Source_Code
* FILTERED_PATH：特定repo上的任务，jsonl，应该是之前的步骤已经生成好了的
* OUTPUT_COMPLETION_PATH：将代码补全的结果存在哪里，jsonl
* METHODS_CSV：RepoSummary提取出的repo中所有method的信息
* ENRE_JSON：对repo做静态分析得到的结果，用于计算context recall时考虑变量是否被包含
* DIAGNOSTIC_JSONL：搜索结果，jsonl
* OUTPUT_COMPLETION_PATH：生成的代码补全文件路径，jsonl
* RAG_DATA_SOURCE：RAG使用的策略，目前有"bm25", "unixcoder", "feature", "feature+bm25"
* GRAPH_DIR_PATH：上一步把图保存在哪个文件夹里，会按照FILTERED_PATH中任务的顺序，依次读取task_1_rank.gml, task_2_rank.gml......，作为context
* GRAPH_PROJECT_PATH：项目根目录（对应于RepoSummary总结的目录），某些时候用来从项目源代码中提取具体的代码内容
* MODEL_NAME：生成代码使用的LLM名称
* MODEL_BACKEND_CHOICE：以什么方式调用LLM，目前有openai方式和ollama方式，ollama方式还没有测试过
* DEBUG：是否打印调试信息，如果为True，会把prompt和LLM的响应结果都打印出来，建议在试新的prompt或策略时，可以打印出来看看
* GENERATION_FLAG：是否做代码生成，默认True。如果只想统计context recall，那么设置为False，就不会真的生成代码，而是只看搜索结果

PROMPT_TEMPLATE是具体的prompt内容，要替换的地方用{{}}来标识。



以上这些，都是针对一个repo进行代码生成。而在批量跑实验的时候，我们需要更方便的运行脚本，比如一次性跑100+repo的代码生成，因此以下几个py文件是用于批量生成的：

* batch_run_no_context.py
* batch_run_base_rag.py
* batch_run_no_context.py

运行的时候，要修改文件开头的路径，尤其注意以下几个，其他的含义与上文相同：

* EXCEL_PATH：excel文件，包含project_name和project_root两列，表示repo名称和repo路径（对应于RepoSummary总结的目录），把需要跑的项目都放在这个excel里
* DEFAULT_BASE_SEARCH_OUT：搜索时生成的一些结果的根目录（这下面就是按不同项目命名的文件夹，比如下面有/mrjob, /boto）
* DEFAULT_DIAGNOSTIC_FILENAME：当使用base RAG方式的时候，从哪个文件里获得搜索结果，比如diagnostic_bm25_code.jsonl
* GRAPH_SUBDIR：在具体的repo文件夹下，图存放在哪个子目录，比如`graph_results_***all/PageRank-15-subgraph`
* DEFAULT_BASE_ENRE：存放enre report的根目录，通常与DEFAULT_BASE_SEARCH_OUT一样
* DEFAULT_BASE_COMPLETION_OUT：生成代码存放的根目录（这下面是/mrjob, /boto）
* COMPLETION_FILENAME：生成的补全代码jsonl的文件名，比如bm25_rag_completion.jsonl
* SUBFOLDER：生成的代码具体存放在repo文件夹下的哪个子目录，比如mrjob下面的0303_full

注意excel表格里是这次要跑代码生成的repo，改好后直接运行即可

生成的completion.jsonl文件里，每一行一定有namespace和completion字段，namespace用来区分是哪个任务，completion是LLM补全出来的代码，已经从LLM的response中截取出来了。









