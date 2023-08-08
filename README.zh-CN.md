<img src="./assets/icon.png" width="18"> [English](./README.md) | 简体中文

# Chart2Vec:  上下文感知的通用可视化嵌入模型

Chart2Vec 是可视化领域内的图表嵌入模型，用于将可视化图表直接转化为向量。经由该模型生成的向量不仅包含单个图表本身的有效信息，同时也涵盖图表间的上下文关系。“上下文”指的是图表的相邻关系，通常存在于多图表可视化（仪表盘、数据故事等）。Chart2Vec 可以作为一种新的数据格式进行存储和计算，并应用到各种可视化下游任务中: 可视化推荐、聚类、插值等等。
<div align=center>
    <img src="assets/model_architecture.png" width="30%" align="middle">
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
* **_testing_data.json_**: 模型的测试数据集，包含 551 条测试样本。每条样本是单图表可视化的声明式语法数据，其中每个图表以 fact_id 进行标识，由三个数字组成，例如"281-816-1"表示281号数据集-816号数据故事/仪表盘-第一幅可视化。
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

已训练完的 Chart2Vec 模型存放路径为：models/chart2vec_base.pth。

## Installation

采用 pip 从文件 requirements.txt 中安装所需的 Python 包：

```shell
pip install -r requirements.txt
```

## Usage

**1. 模型的直接利用：将图表表示为向量**。输入图表的七元组格式，输出为 300 维的向量。

```python
from utils.data_loader import *
from encoder.modeling_chart2vec import *

# 加载 Chart2Vec 模型
chart2vec_model = Chart2Vec().to(device)
path = os.path.dirname(__file__)
model_save_path = os.path.join(
    path, "models/chart2vec_base.pth")
state_dict = torch.load(model_save_path, map_location=torch.device(device))
chart2vec_model.load_state_dict(state_dict['model'])
chart2vec_model.eval()

# 准备图表 7 元组作为输入
chart_fact = {"fact_type": "trend", "subspace": [], "breakdown": [{"field": "Year of release", "type": "temporal"}], "measure": [
    {"field": "Rating", "aggregate": "avg", "type": "numerical"}], "focus": [], "chart_type": "Area Chart", "meta": "increasing"}
# 数据的对齐（多个图表放置在一个list中也是支持的）
input_chart = pad_w2v_convert_fact_to_input([chart_fact])

# 利用 Chart2Vec 将图表表示为向量
output_vec = chart2vec_model(
    np.array([input_chart["batch_struct_one_hot"][0]]),
    np.array([input_chart["batch_indexed_tokens"][0]]),
    np.array([input_chart["batch_pos"][0]]))
# 输出
print(output_vec[0].tolist())
```

**2. 模型的训练：基于新数据集重新训练模型**。正式训练之前，可以按照论文中的描述以及参照 **_dataset/training_data.json_** 文件构建训练数据集。下面我们给出示例代码：

```python
import torch
import json
from utils.data_loader import *
from encoder.modeling_chart2vec import *
from encoder.ImprovedQuadrupletLoss import *

def get_chart2vec_embeddings(chart2vec_model, chart_data):
    return chart2vec_model(chart_data["batch_struct_one_hot"], chart_data["batch_indexed_tokens"], chart_data["batch_pos"])

# 模型的超参数设置可以根据实际情况进行调整
epochs = 50
batch_size = 128
lr = 1e-2

# 设置模型运行的机器
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载训练数据以及构建 batch
with open("./dataset/training_data.json") as f:
        train_data = json.load(f)
batch_gen = quadruplet_batch_generator_word2vec(train_data, batch_size)

# 多任务损失函数
loss_fc = ImprovedQuadrupletLoss().to(device)
chart_emb = Chart2Vec().to(device)
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, chart_emb.parameters()), lr=lr)

# 迭代训练模型
best_loss = 1e7
for i in range(epochs):
    total_loss = 0.0
    for j in range(len(train_data)//batch_size):
        # 将每一条训练样本中的图表表示为向量
        batch_chart_x1, batch_chart_x2, batch_chart_x3, batch_chart_y1 = next(batch_gen)
        res_chart_x1 = get_chart2vec_embeddings(chart_emb,batch_chart_x1)
        res_chart_x2 = get_chart2vec_embeddings(chart_emb,batch_chart_x2)
        res_chart_x3 = get_chart2vec_embeddings(chart_emb,batch_chart_x3)
        res_chart_y1= get_chart2vec_embeddings(chart_emb,batch_chart_y1)
        # 计算损失函数
        chart2vec_loss = loss_fc(res_chart_x1,  res_chart_x2, res_chart_x3, res_chart_y1)
        total_loss += chart2vec_loss.cpu().item()
        # 反向传播以进行优化
        chart2vec_loss.backward()
        optimizer.step()
        chart_emb.zero_grad()
        print("mean_loss:", chart2vec_loss.item())

    # 模型的保存
    if total_loss < best_loss:
        best_loss = total_loss      
        torch.save({'model': chart_emb.state_dict()}, os.path.join("YOUR_MODEL_SAVE_PATH/chart2vec_base.pth"))
```

**3. 模型的验证：检验Chart2Vec模型捕捉上下文关系的能力**。

**评估指标**: 我们分别使用 top-2 retrieval accuracy, top-3 retrieval accuracy 和 co-occurrence 这三个指标衡量 Chart2Vec 嵌入的能力，以评估其是否能有效捕捉到多图表之间的上下文关系。

* **top-2 retrieval accuracy**. 对于锚定图表，根据图表向量之间的距离搜索与之最近的向量所代表的图表，如果两个图表源自统一数据故事，并间距在2以内，则该锚定图表在这一指标上是符合规定的。计算测试集 560 个图表的所有结果，将符合规定的图表个数除以总数，即为最终的值。
* **top-3 retrieval accuracy**.  同top-2 retrieval accuracy的计算方法类似，如果检索到的图表和锚定图表源自统一数据故事，并间距在3以内，则该锚定图表在这一指标上是符合规定的。
* **co-occurrence**. 如果检索到的图表和锚定图表源自统一数据故事，则该锚定图表在co-occurrence上是符合规定的。最终计算所有的符合条件的图表个数，并除以总数可以得到该指标最终的值。

下面我们给出验证模型捕捉图表间的上下文能力的示例代码：

```python
import os
import json
import torch
from tqdm import tqdm
import numpy
from utils.data_loader import *
from utils.constants import *
from encoder.modeling_chart2vec_word_max_pooling import *


def test_and_evaluate_chart2vec(trained_model_path, testing_data_file):
    """
        test and evaluate the chart2vec model
        Inputs:
            `model_save_path`: The directory where model iterations are saved during the training process, the overall directory is located under models.
            `training_data_file`: Filename of the training data.
    """
    path = os.path.dirname(__file__)
    with open(os.path.join(path, testing_data_file)) as f:
        test_data = json.load(f)

    chart2vec_model = Chart2Vec().to(device)
    model_save_path = os.path.join(
        path, "models", trained_model_path+"chart2vec_base.pth")
    state_dict = torch.load(model_save_path, map_location=torch.device(device))
    chart2vec_model.load_state_dict(state_dict['model'])
    chart2vec_model.eval()

    input = pad_w2v_convert_fact_to_input(test_data)
    batch_struct_one_hot = input["batch_struct_one_hot"]
    batch_indexed_tokens = input["batch_indexed_tokens"]
    batch_pos = input["batch_pos"]
    input_len = len(batch_struct_one_hot)

    result_dict = {}
    for i in tqdm(range(input_len), desc="data_num"):
        output_vec = chart2vec_model(
            np.array([batch_struct_one_hot[i]]), np.array([batch_indexed_tokens[i]]), np.array([batch_pos[i]]))
        result_dict[test_data[i]["fact_id"]] = output_vec[0].tolist()

    acc2, acc3, acc_co = evaluate_more_nearest_dis_triplets(
        result_dict, search_num=1)
    print(acc2, acc3, acc_co)


def get_models_folder(dir_name):
    models_folder_list = os.listdir(dir_name)
    for folder in models_folder_list:
        if os.path.isfile(os.path.join(dir_name, folder)):
            models_folder_list.remove(folder)
    return models_folder_list


def evaluate_more_nearest_dis_triplets(data, search_num=1):
    """
        根据模型生成的向量验证 Chart2Vec 的建模结果。指定一个图表，计算离它最近的 top k 个图表，确定它是否位于上下文窗口中，并计算上下文距离。
    """
    same_dataset_facts, accuracy_list_window2, accuracy_list_window3, accuracy_list_story = {}, {}, {}, {}
    last_dataset_key = ""
    for key in list(data.keys()):
        dataset_label = key.split("-", 1)[0]
        if len(same_dataset_facts) == 0 or (len(same_dataset_facts) > 0 and dataset_label == list(same_dataset_facts.keys())[0].split("-", 1)[0]):
            same_dataset_facts[key] = data[key]
            last_dataset_key = key.split("-", 1)[0]
        else:
            # 首先计算上一个数据集中各 facts 之间的准确性
            accuracy_list_window2[last_dataset_key],  accuracy_list_window3[last_dataset_key], accuracy_list_story[last_dataset_key] = cal_same_dataset_facts_more_dis_min(
                same_dataset_facts, search_num)
            same_dataset_facts = {}
            same_dataset_facts[key] = data[key]
    accuracy_list_window2[key], accuracy_list_window3[key], accuracy_list_story[key] = cal_same_dataset_facts_more_dis_min(
        same_dataset_facts, search_num)
    return cal_avg_accuracy_value(accuracy_list_window2), cal_avg_accuracy_value(accuracy_list_window3), cal_avg_accuracy_value(accuracy_list_story)


def cal_avg_accuracy_value(accuracy_list):
    """
        计算整体的平均值
    """
    accuracy_list = dict(sorted(accuracy_list.items(),
                         key=lambda d: d[1], reverse=True))
    return numpy.mean(list(accuracy_list.values()))


def cal_same_dataset_facts_more_dis_min(facts_dict, search_num):
    """
        基于 anchor 图表，计算距离最近的图表
    """
    dis_map = {}
    facts_dict_key = list(facts_dict.keys())
    for i in range(len(facts_dict)):
        temp_dis_map = {}
        key1 = facts_dict_key[i]
        fact1_id = key1.split("-", 1)[1]
        value1 = numpy.array(facts_dict[key1])
        for j in range(len(facts_dict)):
            if i == j:
                continue
            key2 = facts_dict_key[j]
            fact2_id = key2.split("-", 1)[1]
            value2 = numpy.array(facts_dict[key2])
            dis = eucliDist(value1, value2)
            if len(temp_dis_map.keys()) < search_num:
                temp_dis_map[fact2_id] = dis
            else:
                # 如果已经存储了数值，则查找最大值，如果比它小，则替换它。
                temp_max_key = max(temp_dis_map, key=temp_dis_map.get)
                if dis < temp_dis_map[temp_max_key] and dis > 0:
                    del temp_dis_map[temp_max_key]
                    temp_dis_map[fact2_id] = dis
        dis_map[fact1_id] = temp_dis_map
    is_context, is_context_window2, is_context_window3, is_context_story = 0, 0, 0, 0
    for key in dis_map.keys():
        fact1_id = key
        for fact2 in dis_map[key].keys():
            fact2_id = fact2
            if fact1_id.split("-")[0] != fact2_id.split("-")[0]:
                continue
            if int(fact1_id.split("-")[1])+1 == int(fact2_id.split("-")[1]) or int(fact1_id.split("-")[1])-1 == int(fact2_id.split("-")[1]):
                is_context += 1
                is_context_window2 += 1
                is_context_window3 += 1
                is_context_story += 1
                break
            if int(fact1_id.split("-")[1])+2 == int(fact2_id.split("-")[1]) or int(fact1_id.split("-")[1])-2 == int(fact2_id.split("-")[1]):
                is_context_window2 += 1
                is_context_window3 += 1
                is_context_story += 1
                break
            if int(fact1_id.split("-")[1])+3 == int(fact2_id.split("-")[1]) or int(fact1_id.split("-")[1])-3 == int(fact2_id.split("-")[1]):
                is_context_window3 += 1
                is_context_story += 1
                break
            if int(fact1_id.split("-")[0]) == int(fact2_id.split("-")[0]):
                is_context_story += 1
                break
    window2_accuracy = float(is_context_window2/len(dis_map.keys()))
    window3_accuracy = float(is_context_window3/len(dis_map.keys()))
    story_accuracy = float(is_context_story/len(dis_map.keys()))
    return window2_accuracy, window3_accuracy, story_accuracy


def eucliDist(A, B):
    return numpy.sqrt(sum(numpy.power((A - B), 2)))


if __name__ == "__main__":
    test_and_evaluate_chart2vec(trained_model_path="models/chart2vec_base.pth", testing_data_file="dataset/testing_data_50.json")
```
