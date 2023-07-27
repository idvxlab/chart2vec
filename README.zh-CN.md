<img src="./assets/icon.png" width="18"> [English](./README.md) | 简体中文

# Chart2Vec:  上下文感知的通用可视化嵌入模型

Chart2Vec 是可视化领域内的图表嵌入模型，用于将可视化图表直接转化为向量。经由该模型生成的向量不仅包含单个图表本身的有效信息，同时也涵盖图表间的上下文关系。“上下文”指的是图表的相邻关系，通常存在于多图表可视化（仪表盘、数据故事等）。Chart2Vec 可以作为一种新的数据格式进行存储和计算，并应用到各种可视化下游任务中: 可视化推荐、聚类、插值等等。
<div align=center>
    <img src="assets/model_architecture.png" width="50%" align="middle">
    <div>Chart2Vec 模型架构</div>
</div>

Chart2Vec 主要由两大核心模块构成：Input embedding 和 Encoder。该仓库包含 Chart2Vec 模型的核心代码，各文件夹/文件主要的功能如下：

* **_dataset/_**: 训练和测试数据集
* **_input embedding/_**: 将图表的 chart fact 数据表示为初始向量
* **_encoder/_**: 将图表的初始向量进一步编码，融合结构层面和语义层面两部分信息
* **_utils/_**: 数据加载方式、常量的定义和一些帮助类代码
* **_train.py_**: 模型的训练代码
* **_test.py_**: 模型的推理代码
* **_evaluate.py_**: 模型的效果验证代码

## dataset

 _dataset/_ 文件夹中主要包含模型的训练数据集和测试数据集，这些数据集源自数据故事平台 [Calliope](https://datacalliope.com/) 和 [Tableau Public](https://public.tableau.com/)

* **_training_data.json_**: 模型的训练数据集，包含 42,222 条训练样本。每条训练样本由 4 个可视化构成，前三个为同一个数据故事/仪表盘顺次相连的可视化，第四个为负例（和前三个图表不在同一数据故事/仪表盘中）。
* **_testing_data.json_**: 模型的测试数据集，包含 560 条测试样本。每条样本是单图表可视化的声明式语法数据，其中每个图表以 fact_id 进行标识，由三个数字组成，例如"281-816-1"表示281号数据集-816号数据故事/仪表盘-第一幅可视化。
* **_story_stopwords.txt_**: 停用词词表。

**注意**：我们采用 chart fact 作为单个可视化图表的初始格式，它由 7 元组构成：chart fact={fact_type, chart_type, subspace, breakdown, measure, focus, meta}，其中部分字段定义源自 [Calliope](https://ieeexplore.ieee.org/document/9222368)， 7 元组详细解析如下表所示：
|  Property   | Type  | Description |
|  ----  | ----  | ----  |
| fact_type  | String | 数据事实类型，可选项有 10 种： `"trend"`, `"categorization"`, `"value"`, `"difference"`, `"distribution"`, `"proportion"`, `"rank"`, `"extreme"`, `"outlier"`, `"association"` |
| chart_type  | String | 图表类型，可选项有 18 种： `"Vertical Bar Chart"`, `"Pie Chart"`, `"Progress Bar Chart"`, `"Treemap"`, `"Line Chart"`, `"Text Chart"`, `"Area Chart"`, `"Horizontal Bar Chart"`, `"Proportion Isotype Chart"`, `"Scatter Plot"`, `"Color Filling Map"`, `"Bubble Chart"`, `"Ring Chart"`, `"Bubblemap"`, `"Isotype Bar Chart"`, `"Table"`, `"Network"`, `"Radar Chart"`  |
| subspace  | Object[] | 由一组过滤器构成，用于筛选数据范围 |
| breakdown  | Object[] | 由一组时间或分类类型的数组字段构成，进一步将子空间的数据项划分为组 |
| measure  | Object[] | 数值数据字段，可结合不同的聚合方法对分组中的数据进一步度量 |
| focus  | Object[] | 需要注意的数据项或数据组 |
| meta  | [] \| "" | 图表的额外信息，根据不同的 fact type 具有不同的信息 |

## input_embedding 和 encoder

* **_input_embedding/semantic_extract.py_**: 从原始图表数据 chart fact 提取语义信息，即提取具有实际意义的单词，进行分词、去除停用词后统一存放在一个数组中。
* **_input_embedding/structual_extract.py_**: 从原始图表数据 chart fact 提取结构信息，表示成 one-hot 向量构成的矩阵。
* **_input_embedding/structual_rules.txt_**: 从 chart fact 可提取的规则信息的全集。
* **_encoder/modeling_chart2vec.py_**: Chart2Vec 核心网络结构，对图表的语义信息和结构信息进行融合。

<div align=center>
    <img src="assets/model_chart_format.png" width="100%" align="middle">
    <div>结构信息和语义信息</div>
</div>

## Chart2Vec model

 你可以下载我们的Chart2Vec模型，并采用该仓库中的代码进行模型的加载和使用：[Chart2Vec](download).

## Usage

* **_train.py_**: 模型的训练, 可直接运行 train.py 代码，以下是示例代码

```python
# 多任务损失函数
loss_fc = ImprovedQuadrupletLoss().to(device)
chart_emb = Chart2Vec().to(device)
for i in range(epochs):
    total_loss = 0.0
    for j in range(batch_num):
        # struct_one_hot_x1 为结构信息，semantic_tokens_x1, semantic_pos_x1 为语义信息
        res_chart_x1 = chart_emb(struct_one_hot_x1, semantic_tokens_x1, semantic_pos_x1)
        res_chart_x2 = chart_emb(struct_one_hot_x2, semantic_tokens_x2, semantic_pos_x2)
        res_chart_x3 = chart_emb(struct_one_hot_x3, semantic_tokens_x3, semantic_pos_x3)
        res_chart_y1 = chart_emb(struct_one_hot_y1, semantic_tokens_y1, semantic_pos_y1)
        chart2vec_loss = loss_fc(res_chart_x1,  res_chart_x2, res_chart_x3, res_chart_y1)
        total_loss += chart2vec_loss.cpu().item()
        chart2vec_loss.backward()
        optimizer.step()
        chart_emb.zero_grad()
```

* **_test.py_**: 模型的推理，将 dataset/testing_data.json 中的每个单图表都表示成了 300 维的向量，以下是示例代码

```python
model_path = "chart2vec_base.pth"
chart2vec_model = Chart2Vec().to(device)
state_dict = torch.load(model_save_path, map_location=torch.device(device))
chart2vec_model.load_state_dict(state_dict["model"])
chart2vec_model.eval()
output_vec = chart2vec_model(struct_one_hot, semantic_tokens, semantic_pos)
```

* **_evaluate.py_**: 模型的评估，我们分别使用 top-2 retrieval accuracy, top-3 retrieval accuracy 和 co-occurrence 这三个指标衡量 Chart2Vec 嵌入的能力，以评估其是否能有效捕捉到多图表之间的上下文关系。

  * **top-2 retrieval accuracy**. 对于锚定图表，根据图表向量之间的距离搜索与之最近的向量所代表的图表，如果两个图表源自统一数据故事，并间距在2以内，则该锚定图表在这一指标上是符合规定的。计算测试集 560 个图表的所有结果，将符合规定的图表个数除以总数，即为最终的值。
  * **top-3 retrieval accuracy**.  同top-2 retrieval accuracy的计算方法类似，如果检索到的图表和锚定图表源自统一数据故事，并间距在3以内，则该锚定图表在这一指标上是符合规定的。
  * **co-occurrence**. 如果检索到的图表和锚定图表源自统一数据故事，则该锚定图表在co-occurrence上是符合规定的。最终计算所有的符合条件的图表个数，并除以总数可以得到该指标最终的值。


```python
# 测试数据集中的每个单图表都表示成向量后的数据
path_name="results/testing_data_vectors.json"
# 针对选定的图表搜索距离最近的1个
evaluate_more_nearest_dis_triplets(path_name,search_num=1)
```