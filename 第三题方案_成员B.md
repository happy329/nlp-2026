

# 第三题：基于Transformer的中文新闻分类与摘要系统

# 一、三人分工

## 成员 A：数据预处理 + 分类模型（训练模型）

### 任务

```text
1. 整理 cnews / THUCNews 数据集
2. 编写文本清洗代码
3. 编写 jieba 分词代码
4. 构建 stopwords 停用词表
5. 构建 label2id / id2label
6. 构建 word2id 词表
7. 实现Word2Vec(CBOW):用于初始化模型的embeddingc层
8. 实现 TextCNN
9. 实现 BiLSTM
10. 实现 BiLSTM-Attention
11. 输出分类实验结果和相关可视化
```

### 交付物

```text
preprocess/jieba_fenci.py
preprocess/build_cibiao.py
preprocess/train_word2vec.py

classification/textcnn.py
classification/bilstm.py
classification/bilstm_attention.py
classification/classifier_api.py

train/train_textcnn.py
train/train_bilstm.py
train/train_bilstm_attention.py

results/classification_metrics.csv
results/figures/loss_curve.png
results/figures/confusion_matrix.png
```

------

## 成员 B：自动摘要 + 关键词提取模块

成员 B 负责两个相对独立的任务：自动摘要和关键词提取。

自动摘要部分使用 LCSTS 中文摘要数据集进行训练与评价。。摘要实验重点比较传统抽取式摘要方法与 Transformer 生成式摘要方法。

关键词提取部分使用 CSL 中文科学文献数据集进行评价。CSL 数据集中包含论文标题、摘要和人工关键词，适合用于关键词提取方法的客观对比。

### 任务1：自动摘要

#### 1.1 实验模型

```text
1. TextRank：
   抽取式摘要基线方法，不需要训练，通过句子相似度建图和 PageRank 排序选取关键句。

2. Pegasus-base zero-shot：
   加载未经过 LCSTS 摘要任务微调的中文 Pegasus 预训练模型，直接进行摘要生成，用于观察预训练模型在未微调时的摘要能力。

3. Pegasus-base + LCSTS 微调：
   使用 LCSTS 训练集对中文 Pegasus 预训练模型进行摘要任务微调，保存成员 B 自己训练得到的摘要模型。

4. Randeng-Pegasus-238M-Summary-Chinese：
   直接调用已经在中文摘要数据集上微调好的公开摘要模型
   IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese，
   作为较强的生成式摘要对比模型。
```

#### 1.2 数据集

```text
LCSTS 中文摘要数据集

训练集：用于微调 Pegasus-base
验证集：用于调参和选择模型
测试集：用于比较 TextRank、Pegasus-base zero-shot、自己微调 Pegasus、公开微调 Pegasus 的摘要效果
```

#### 1.3 评价指标

```text
1. ROUGE-1
2. ROUGE-2
3. ROUGE-L
4. BERTScore
5. 摘要长度
6. 压缩率
```

其中：

```text
ROUGE 用于衡量生成摘要与人工摘要之间的字面重合程度；
BERTScore 用于衡量生成摘要与人工摘要之间的语义相似度；
摘要长度和压缩率用于比较不同方法的压缩能力。
```

#### 1.4 自动摘要流程

```text
LCSTS 正文
    ↓
TextRank / Pegasus-base / 自己微调 Pegasus / 公开微调 Pegasus
    ↓
生成摘要
    ↓
与 LCSTS 人工摘要比较
    ↓
计算 ROUGE、BERTScore、摘要长度和压缩率
    ↓
输出摘要实验结果表和对比图
```

#### 1.6 自动摘要交付物

```text
summarization/prepare_lcsts.py：整理 LCSTS 数据集，构造 content-summary 样本
summarization/train_summary_model.py：使用 LCSTS 微调中文 Pegasus
summarization/pegasus_summary.py：加载 Pegasus 模型并生成摘要
summarization/textrank_summary.py：实现 TextRank 抽取式摘要
summarization/summary_api.py：摘要统一接口

evaluate/eval_summary.py：计算 ROUGE、BERTScore、摘要长度和压缩率


results/rouge_score.csv：ROUGE 评价结果
results/bertscore.csv：BERTScore 评价结果
results/summary_length.csv：摘要长度与压缩率统计

saved_models/summary_model/：保存自己微调后的 Pegasus 摘要模型
```

### 任务2：关键词提取

#### 2.1 实验方法

```text
1. TF-IDF：
   基于词频和逆文档频率提取关键词，作为传统统计方法基线。

2. TextRank：
   基于词语共现图和 PageRank 排序提取关键词，作为传统图排序方法。

3. KeyBERT：
   使用句向量模型计算文档向量与候选关键词向量之间的语义相似度，选择与文档语义最接近的关键词。
```

#### 2.2 数据集

```text
CSL 中文科学文献数据集

输入：论文标题 + 摘要
标准答案：论文作者给出的关键词
任务：从标题和摘要中自动提取 Top-K 关键词，并与人工关键词比较
```

#### 2.3 评价指标

```text
1. Precision@5、Recall@5、F1@5
2. Precision@10、Recall@10、F1@10
```

#### 2.4 关键词提取流程

```text
CSL 标题 + 摘要
    ↓
文本清洗
    ↓
候选词生成
    ↓
TF-IDF / TextRank / KeyBERT 提取关键词
    ↓
与 CSL 人工关键词比较
    ↓
计算 P@K、R@K、F1@K
    ↓
输出关键词实验结果表和对比图
```

#### 2.5 关键词提取交付物

```text
summarization/keyword_api.py：关键词提取统一接口
summarization/tfidf_keywords.py：TF-IDF 关键词提取
summarization/textrank_keywords.py：TextRank 关键词提取
summarization/keybert_keywords.py：KeyBERT 关键词提取

evaluate/eval_keywords.py：计算 P@K、R@K、F1@K

results/keyword_scores.csv：关键词提取评价结果
results/figures/keyword_method_compare.png：不同关键词方法指标对比图
```


------

## 成员 C：相似新闻推荐 + 系统界面（不训练模型）

推荐用到的数据集使用news.csv，内容包含train/test/val.csv

### 任务

```text
1. 实现 TF-IDF + 余弦相似度推荐（可选，可以做对比）
2. 实现 Sentence-BERT（直接使用） + 余弦相似度推荐
3. 根据分类模块输出结果，优先进行同类别新闻推荐
4. 编写推荐模块统一接口 recommend_api.py
5. 搭建 Streamlit 系统界面 app.py
6. 调用分类、摘要、推荐接口，完成系统整合
7. 绘制推荐结果与系统展示相关可视化图表
```

### 交付物

```text
recommendation/tfidf_recommender.py(可选)
recommendation/sbert_recommender.py :直接使用sbert模型
recommendation/build_index.py
recommendation/recommend_api.py：推荐统一接口

pipeline.py：系统总接口，统一调用分类模块、关键词模块、摘要模块和推荐模块，返回完整分析结果
app.py：Streamlit 系统界面，负责接收用户输入新闻，并展示分类结果、关键词、自动摘要和相似新闻推荐结果

visualization/plot_category_distribution.py：绘制新闻库类别分布图，用于展示 cnews / news.csv 中各类别新闻数量
visualization/plot_similarity.py：绘制Top-K推荐新闻相似度柱状图
visualization/plot_embedding.py：对 Sentence-BERT 新闻向量进行降维可视化，展示不同类别新闻在语义空间中的分布情况

results/recommend_examples.csv
results/figures/similarity_topk.png
results/figures/category_distribution.png
results/figures/embedding_visualization.png

saved_index/tfidf_vectorizer.pkl（可选）：保存 TF-IDF 向量化器，用于把用户输入新闻转换成 TF-IDF 向量。
saved_index/tfidf_matrix.pkl（可选）：保存新闻库的 TF-IDF 向量矩阵，用于 TF-IDF 推荐。
saved_index/sbert_embeddings.npy：保存 news.csv 中所有新闻的 Sentence-BERT 语义向量，用于快速计算相似度。
saved_index/news_ids.json：保存新闻向量与 news.csv 中 news_id 的对应关系，方便根据推荐结果回查新闻正文、类别等信息。
```

build_index.py：这个文件负责提前构建推荐索引。

作用是：

```
读取 news.csv
    ↓
提取 news_id / content / label
    ↓
构建 TF-IDF 矩阵
    ↓
构建 Sentence-BERT 向量
    ↓
保存到 saved_index/
```

这样系统运行时不用每次重新计算全部新闻向量。

系统运行时只需要：

```
加载 saved_index，输入新闻转向量，计算相似度，返回 Top-K
```

TF-IDF 推荐：
新闻文本 → jieba分词 → TF-IDF向量 → 余弦相似度 → Top-K推荐

Sentence-BERT 推荐：
新闻文本 → Sentence-BERT编码 → 句向量 → 余弦相似度 → Top-K推荐

# 二、数据格式统一(必读)

所有**路径**和**超参数**都写在config.py下，方便管理，在对应文件下通过`import config as cfg`引入

### data/processed

成员A和C用cnews分类数据集，即在data/processed下

成员B的自动摘要实验使用 LCSTS 摘要数据集，放在 `data/LCSTS/` 下；关键词提取实验使用 CSL 中文科学文献数据集，放在 `data/CSL/` 下。

```
data/processed/
├── hit_stopwords.txt（哈工大停用词表）
├── label2id.json
├── id2label.json

├── train.csv
├── val.csv
├── test.csv
├── news.csv（合并letrain+val+test，用于新闻推荐的输入）

├── train_fenci.csv
├── val_fenci.csv
├── test_fenci.csv
├── news_fenci.csv

├── vocab.json
└── vocab.txt
```

## 1.文件格式

### 1.1 train.csv / val.csv / test.csv/news.csv 格式

统一字段：

```text
news_id, content , label , label_id
```

### 1.2 news.csv 格式(同train.csv)

`news.csv` 是全量新闻库，由 train / val / test 合并得到。

作用：

```text
1. 推荐模块构建新闻库
2. 可视化统计类别分布
3. 系统演示时随机抽样新闻
```

### 1.3 train/val/test_fenci.csv 格式

```
news_id , cut_text , label , label_id
train_000001,马晓旭 意外 受伤 国奥 警惕,体育,0
train_000002,商瑞华 首战 复仇 中国 玫瑰,体育,0
train_000003,冠军 球队 迎新 欢乐 派对,体育,0
```

Train/val/test_fenci.csv里没有放ids列，因为 `ids` 是根据 `vocab.json` 转出来的，后面词表一变，`ids` 就要重新生成。更合理的是训练时动态转：

```
train.csv / val.csv / test.csv
        ↓
jieba 分词
        ↓
train_fenci.csv / val_fenci.csv / test_fenci.csv
        ↓
根据 train_fenci.csv 构建 vocab.json / vocab.txt
        ↓
训练 TextCNN / BiLSTM 时再把 cut_text 转成 ids
```

### 1.4 LCSTS 数据集

LCSTS 只用于成员 B 的自动摘要实验，放在 `data/LCSTS/` 下。建议整理成统一字段：

```text
sample_id, content, summary
```

其中：

```text
content：原文
summary：人工摘要
```

### 1.5 CSL 中文科学文献数据集

CSL 只用于成员 B 的关键词提取实验，放在 `data/CSL/` 下。建议整理成统一字段：

```text
sample_id, title, abstract, keywords
```

其中：

```text
title：论文标题
abstract：论文摘要
keywords：人工关键词，可用分号或列表格式保存
```

## 2.统一接口

### 2.1 分类接口

文件：

```text
classification/classifier_api.py
```

函数：

```python
predict_class(text, model_name="roberta")
```

返回格式：

```python
{
    "label": "科技",
    "label_id": 3,
    "probs": {
        "体育": 0.01,
        "财经": 0.03,
        "娱乐": 0.02,
        "科技": 0.91,
        "教育": 0.01
    }
}
```

### 2.2 摘要接口

文件：

```text
summarization/summary_api.py
```

函数：

```python
summarize(text, method="both")
```

返回格式：

```python
{
    "generate_summary": "某公司发布新一代AI芯片，面向大模型训练与推理。",
    "textrank_summary": "某公司发布新一代人工智能芯片，该芯片主要面向大模型训练和推理场景。"
}
```

### 2.3关键词接口

文件：

```text
summarization/keyword_api.py
```

或者放在：

```text
preprocess/keyword.py
```

函数：

```python
extract_keywords(text, top_k=10)
```

返回格式：

```python
["人工智能", "芯片", "大模型", "推理", "算力"]
```

### 2.4 推荐接口

文件：

```text
recommendation/recommend_api.py
```

函数：

```python
recommend(text, label_filter=None, top_k=5, method="sbert")
```

返回格式：

```python
[
    {
        "news_id": "000032",
        "title": "国产AI芯片发展加速",
        "label": "科技",
        "score": 0.87
    },
    {
        "news_id": "000145",
        "title": "大模型算力需求增长",
        "label": "科技",
        "score": 0.82
    }
]
```

推荐逻辑：

```text
如果 label_filter 不为空：
    先在该类别下推荐
    如果数量不足 top_k，再从全库补足
如果 label_filter 为空：
    直接全库推荐
```

### 2.5 总系统接口

```text
pipeline.py
```

函数：

```python
run_pipeline(text)
```

返回：

```python
{
    "classification": {
        "label": "科技",
        "label_id": 3,
        "probs": {
            "科技": 0.91,
            "财经": 0.04
        }
    },
    "keywords": ["人工智能", "芯片", "大模型", "推理", "算力"],
    "summary": {
        "textrank_summary": "某公司发布新一代人工智能芯片。",
        "pegasus_summary": "某公司发布AI芯片，面向大模型训练与推理。"
    },
    "recommendations": [
        {
            "news_id": "000032",
            "title": "国产AI芯片发展加速",
            "label": "科技",
            "score": 0.87
        }
    ]
}
```

`app.py` 只调用这个接口，不在 `app.py` 里写模型细节。

## 3.最终目录结构

```
nlp_final/
│
├── data/
│   ├── raw/
│   │   ├── cnews.train.txt
│   │   ├── cnews.val.txt
│   │   └── cnews.test.txt
│   │
│   └── processed/
│   │   ├── train.csv
│   │   ├── val.csv
│   │   ├── test.csv
│   │   ├── news.csv
│   │   ├── label2id.json
│   │   ├── id2label.json
│   │   ├── vocab.json
│   │   └── hit_stopwords.txt
│   ├── LCSTS/
│   └── CSL/
│
├── preprocess/
│   ├── jieba_fenci.py
│   ├── build_cibiao.py
│   ├── pre.py
│   └── keyword.py
	  └── word2vec.py
│
├── classification/
│   ├── textcnn.py
│   ├── bilstm.py
│   ├── bilstm_attention.py
│   └── classifier_api.py
│
├── train/
│   ├── train_textcnn.py
│   ├── train_bilstm.py
│   ├── train_bilstm_attention.py
│
├── summarization/
│   ├── prepare_lcsts.py
│   ├── train_summary_model.py
│   ├── pegasus_summary.py
│   ├── textrank_summary.py
│   ├── summary_api.py
│   ├── tfidf_keywords.py
│   ├── textrank_keywords.py
│   ├── keybert_keywords.py
│   └── keyword_api.py
│
├── evaluate/
│   ├── eval_summary.py
│   └── eval_keywords.py
│
├── recommendation/
│   ├── tfidf_recommender.py
│   ├── sbert_recommender.py
│   ├── build_index.py
│   └── recommend_api.py
│
├── visualization/
│   ├── plot_category_distribution.py
│   ├── plot_metrics.py
│   ├── plot_confusion_matrix.py
│   ├── plot_similarity.py
│   └── plot_embedding.py
│
├── results/
│   ├── classification_metrics.csv
│   ├── summary_examples.csv
│   ├── rouge_score.csv
│   ├── bertscore.csv
│   ├── summary_length.csv
│   ├── keyword_examples.csv
│   ├── keyword_scores.csv
│   ├── recommend_examples.csv
│   └── figures/
│       ├── category_distribution.png
│       ├── model_accuracy.png
│       ├── model_f1.png
│       ├── loss_curve.png
│       ├── confusion_matrix.png
│       ├── summary_compare.png
│       ├── keyword_method_compare.png
│       ├── similarity_topk.png
│       └── embedding_visualization.png
│
├── saved_models/
│   ├── textcnn.pt
│   ├── bilstm.pt
│   ├── bilstm_attention.pt
│   └── summary_model/
│
├── saved_index/
│   ├── tfidf_vectorizer.pkl
│   ├── tfidf_matrix.pkl
│   ├── sbert_embeddings.npy
│   └── news_ids.json
│
├── config.py（配置设置）
├── pipeline.py
├── app.py
└── README.md
```

## 4.config.py统一配置

从 `config.py` 读路径和参数。

```python
# 停用词的路径
stopwords_file = "data/processed/hit_stopwords.txt"

# Word2Vec的路径
word2vec_model_file = "data/processed/word2vec.model"
embedding_matrix_file = "data/processed/embedding_matrix.npy"

word2vec_dim = 200
word2vec_window = 5
word2vec_min_count = 2

# 成员B：自动摘要数据与模型路径
lcsts_dir = "data/LCSTS"
summary_model_dir = "saved_models/summary_model"
public_summary_model = "IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese"

# 成员B：关键词提取数据路径
csl_dir = "data/CSL"
```

# 三、最终项目定位

最终实现这几个功能：

```text
输入一篇中文新闻
    ↓
文本清洗与预处理
    ↓
1.新闻分类：判断新闻属于哪个类别
    ↓
关键词提取：提取新闻核心词
    ↓
2.自动摘要：生成新闻摘要
    ↓
3.相似新闻推荐：推荐相关新闻
    ↓
可视化展示：展示分类概率、摘要结果、相似新闻和实验图表
```

# 四、模块之间关系

分类、摘要、推荐本质上是**三个并列功能模块**，都以同一篇新闻作为输入,但为了让系统逻辑更强，设计成**弱递进关系**：

```text
用户输入新闻
    ↓
统一预处理
    ↓
新闻分类
    ↓
根据分类结果筛选同类别新闻库
    ↓
相似新闻推荐
    ↓
同时生成摘要和关键词
```

| 模块         | 作用                         |
| ------------ | ---------------------------- |
| 新闻分类     | 判断新闻类别                 |
| 自动摘要     | 提取新闻核心内容             |
| 相似新闻推荐 | 优先在同类别新闻中找相似新闻 |
| 关键词提取   | 提供解释性                   |

举例：

输入新闻：

```text
近日，某科技公司发布新一代人工智能芯片，主要面向大模型训练和推理场景……
```

系统输出：

```text
预测类别：科技

关键词：
人工智能、芯片、大模型、推理、科技公司

TextRank摘要：
某科技公司发布新一代人工智能芯片，该芯片主要面向大模型训练和推理场景。

Pegasus摘要：
某科技公司发布新一代AI芯片，面向大模型训练与推理应用。

相似新闻推荐：
1. 国产AI芯片发展加速
2. 大模型算力需求持续增长
3. 智能计算中心建设进入新阶段
```

这样系统看起来是一整套流程，不是几个孤立小功能。

# 五、数据集

## 1. 做分类用到的数据集(THUCNews)

THUCNews包含70多万新闻，先小数据集跑通，用 cnews （THUCNews的子集）

cnews有10类新闻

------

## 2. 摘要数据集：LCSTS 中文摘要数据集

成员 B 的自动摘要实验使用 LCSTS。

LCSTS 数据格式是：

```text
正文 content → 人工摘要 summary
```

用途：

```text
1. 训练：用于微调 Pegasus-base
2. 验证：用于调参和选择模型
3. 测试：用于比较 TextRank、Pegasus-base zero-shot、自己微调 Pegasus、公开微调 Pegasus
```

自动摘要实验评价以 LCSTS 为准，因为 LCSTS 提供人工摘要标签。

------

## 3. 关键词提取数据集：CSL 中文科学文献数据集

成员 B 的关键词提取实验使用 CSL。

CSL 数据格式是：

```text
论文标题 title + 论文摘要 abstract → 人工关键词 keywords
```

用途：

```text
1. 比较 TF-IDF、TextRank、KeyBERT 三种关键词提取方法
2. 计算 Precision@K、Recall@K、F1@K、MAP@10
3. 输出关键词提取结果表和方法对比图
```

关键词提取实验评价以 CSL 为准，因为 CSL 提供人工关键词。

------

## 4. 推荐模块数据

不需要额外数据集，用news.csv

推荐时：

```text
输入新闻 → 计算它和新闻库中其他新闻的相似度 → 返回 Top-K
```

如果已经预测出类别，就先筛选同类别新闻：

```text
输入新闻预测为“科技”
    ↓
只在科技类新闻中计算相似度
    ↓
返回科技类相似新闻
```

------

# 六、预处理模块(A)

输入一个新闻文本，系统应先经过清洗、分词、编码、句子切分，再进入分类、摘要和推荐模块。

## 1. 文本清洗

输入原始新闻后，先做清洗：

```text
去除多余空格，去除换行符，去除无意义符号，过滤过短文本，去重
```

------

## 2. 标签映射

分类模型不能直接使用中文标签，需要转成数字。

```python
label2id = {
    "体育": 0,"财经": 1,"娱乐": 2, "科技": 3,"教育": 4}

id2label = {
    0: "体育",1: "财经",2: "娱乐",3: "科技",4: "教育"}
```

模型训练时：科技 → 3        模型预测时：3 → 科技

------

## 3. 做两条预处理路线

不同模型需要不同输入，有的模型需要以分词作为输入，有的模型不需要，所以要对需要用jieba的模型做jieba

### 路线一：jieba 分词路线（用jieiba的模型需要做的）

用于：

```text
TF-IDF （用于推荐模型）
TextCNN（用于分类模型）
BiLSTM（用于分类模型）
BiLSTM-Attention（用于分类模型）
TextRank摘要
关键词提取
TF-IDF推荐
词云图
Attention热力图
```

流程：

```text
原始新闻
    ↓
文本清洗
    ↓
jieba分词
    ↓
去停用词
    ↓
构建词表 / TF-IDF向量 / 词序列
```



------

### 路线二：Tokenizer 编码路线（不用jieiba的模型需要做的）

```text
Pegasus
Sentence-BERT
```

流程：

```text
原始新闻
    ↓
文本清洗
    ↓
模型自带 tokenizer
    ↓
input_ids / attention_mask
    ↓
模型输入
```

例如 BERT/RoBERTa 输入：

```text
[CLS] 人 工 智 能 芯 片 产 业 正 在 快 速 发 展 [SEP]
```

所以预处理模块应该同时保存：

```text
tokens：jieba分词结果
ids：TextCNN/BiLSTM用的词编号序列
bert_input：RoBERTa tokenizer编码结果
```

------

## 4. 词表构建

TextCNN、BiLSTM、Attention-LSTM 需要词表。

```python
word2id = {
    "<PAD>": 0,
    "<UNK>": 1,
    "人工智能": 2,
    "芯片": 3,
    "产业": 4,
    "发展": 5
}
```

新闻分词后：

```text
["人工智能", "芯片", "产业", "发展"]
```

转成：

```text
[2, 3, 4, 5]
```

------

## 5. Padding 和截断

神经网络输入长度要统一。

建议：

```text
max_len = 200 或 256
```

短文本补 0：

```text
[2, 3, 4, 5, 0, 0, 0, ...]
```

长文本截断：

```text
只保留前 200 / 256 个词
```

BERT/RoBERTa 可以设置：

```text
max_length = 256 或 512
```

## 6.Word2Vec

作用：让模型的embedding层一开始有更合理的初始化词向量

```
train_fenci.csv
    ↓
build_cibiao.py 生成 vocab.json
    ↓
新增 train_word2vec.py 训练 Word2Vec
    ↓
生成 word2vec.model 和 embedding_matrix.npy
    ↓
TextCNN / BiLSTM / BiLSTM-Attention 的 Embedding 层用它初始化
```

用什么类型？

CBOW：建议

Skip-gram:对低频词敏感，训练更慢

------

# 七、分类模型(A)

## 1. TextCNN

作用：用 CNN 提取局部短语特征。

结构：

```text
输入词序列
    ↓
Embedding
    ↓
多尺度卷积核：2 / 3 / 4 / 5
    ↓
Max Pooling
    ↓
拼接特征
    ↓
全连接层
    ↓
Softmax分类
```

它适合抓新闻里的关键词组：

```text
亚洲杯、人工智能芯片、央行政策、电影票房
```

优点：

```text
训练快
结构简单
分类效果通常不错
适合作为深度学习基线
```

------

## 2. BiLSTM

作用：从前后两个方向读文本，建模上下文。

结构：

```text
输入词序列
    ↓
Embedding
    ↓
正向 LSTM
    ↓
反向 LSTM
    ↓
拼接双向隐藏状态
    ↓
全连接层
    ↓
Softmax分类
```

它比 TextCNN 更关注顺序和上下文。

例子：

```text
苹果公司发布新款手机
```

BiLSTM 看到“发布新款手机”，就能判断“苹果”是公司，不是水果。

------

## 3. BiLSTM-Attention

作用：在 BiLSTM 后面加入 Attention，让模型自动找重点词。

结构：

```text
输入词序列
    ↓
Embedding
    ↓
BiLSTM
    ↓
Attention层
    ↓
加权句向量
    ↓
Softmax分类
```

这个模型适合做可解释展示。

比如一篇科技新闻，Attention 权重较高的词可能是：

```text
人工智能、芯片、大模型、推理、算力
```

可视化时可以画成：

```text
人工智能 0.22
芯片     0.18
大模型   0.15
推理     0.10
```

------

# 八、自动摘要模块(B)

## 1. 自动摘要任务定义

自动摘要的目标是将较长文本压缩成较短文本。本课设成员 B 的摘要实验只在 LCSTS 数据集上进行，因为 LCSTS 提供了“正文-人工摘要”样本，可以用于训练和自动评价。

自动摘要方法分为两类：

```text
抽取式摘要：从原文中选择关键句组成摘要，例如 TextRank。
生成式摘要：模型阅读原文后重新生成摘要，例如 Pegasus。
```

## 2. TextRank 抽取式摘要基线

TextRank 不需要训练，适合作为传统摘要基线。

流程：

```text
LCSTS 正文
    ↓
分句
    ↓
jieba 分词
    ↓
计算句子相似度
    ↓
构建句子图
    ↓
PageRank 排序
    ↓
选取得分最高的句子
    ↓
组成摘要
```

特点：

```text
优点：稳定，不需要训练，结果来自原文
缺点：只能抽取原文句子，不能主动压缩和改写
```

## 3. Pegasus 生成式摘要模型

Pegasus 属于 Encoder-Decoder Transformer 生成模型，适合自动摘要任务。

本课设比较三种 Pegasus 使用方式：

```text
1. Pegasus-base zero-shot：
   不使用 LCSTS 微调，直接生成摘要。

2. Pegasus-base + LCSTS 微调：
   使用 LCSTS 训练集进行摘要任务微调，得到自己训练的摘要模型。

3. Randeng-Pegasus-238M-Summary-Chinese：
   直接调用公开的中文摘要微调模型，作为较强生成式摘要对比模型。
```

## 4. 自动摘要实验评价

使用 LCSTS 测试集进行评价。

评价指标：

```text
ROUGE-1
ROUGE-2
ROUGE-L
BERTScore
摘要长度
压缩率
```


## 5. 摘要统一接口

文件：

```text
summarization/summary_api.py
```

函数：

```python
summarize(text, method="both")
```

返回格式：

```python
{
    "generate_summary": "生成式摘要结果",
    "textrank_summary": "TextRank抽取式摘要结果"
}
```

------

# 九、关键词提取模块(B)

## 1. 关键词提取任务定义

关键词提取的目标是从文本中抽取最能代表文本主题的词语或短语。本课设成员 B 的关键词实验只使用 CSL 中文科学文献数据集，因为 CSL 提供人工关键词，可以做客观评价。

CSL 输入与标签：

```text
输入：论文标题 + 论文摘要
标签：人工关键词
```

## 2. TF-IDF 关键词提取

TF-IDF 根据词频和逆文档频率计算词的重要性。

流程：

```text
标题 + 摘要
    ↓
分词
    ↓
停用词过滤
    ↓
计算 TF-IDF 权重
    ↓
输出 Top-K 关键词
```

特点：

```text
优点：简单、快速、稳定
缺点：主要依赖词频统计，语义表达能力有限
```

## 3. TextRank 关键词提取

TextRank 根据词语共现关系构建图，并用 PageRank 思想对词语排序。

流程：

```text
标题 + 摘要
    ↓
分词
    ↓
构建词语共现图
    ↓
PageRank 排序
    ↓
输出 Top-K 关键词
```

特点：

```text
优点：不需要训练，能利用词语共现结构
缺点：对分词质量和文本长度较敏感
```

## 4. KeyBERT 关键词提取

KeyBERT 使用句向量模型得到文档向量和候选关键词向量，通过余弦相似度选择与文档语义最接近的关键词。

流程：

```text
标题 + 摘要
    ↓
生成文档向量
    ↓
生成候选关键词向量
    ↓
计算余弦相似度
    ↓
输出 Top-K 关键词
```

特点：

```text
优点：能利用语义相似度，结果通常更贴近文本主题
缺点：推理速度比 TF-IDF 和 TextRank 慢，需要加载预训练句向量模型
```

## 5. 关键词提取评价

评价数据集：

```text
CSL 中文科学文献数据集
```

评价指标：

```text
Precision@5、Recall@5、F1@5
Precision@10、Recall@10、F1@10
MAP@10（可选）
```

## 6. 关键词统一接口

文件：

```text
summarization/keyword_api.py
```

函数：

```python
extract_keywords(text, top_k=10, method="tfidf")
```

返回格式：

```python
["深度学习", "文本分类", "注意力机制", "预训练模型", "自然语言处理"]
```

------

# 十、相似新闻推荐模块(C)

推荐模块做两个方法：

```text
TF-IDF推荐
Sentence-BERT推荐
```

## 1. TF-IDF + 余弦相似度

流程：

```text
新闻库
    ↓
jieba分词
    ↓
TF-IDF向量化
    ↓
输入新闻也转成TF-IDF向量
    ↓
计算余弦相似度
    ↓
返回Top-K
```

## 2. Sentence-BERT + 余弦相似度

流程：

```text
新闻库
    ↓
Sentence-BERT编码成语义向量
    ↓
输入新闻编码成语义向量
    ↓
计算余弦相似度
    ↓
返回Top-K
```

## 3. 推荐时结合分类结果

推荐逻辑建议写成：

```python
label = predict_class(text)
recommend(text, label_filter=label, top_k=5)
```

也就是：

```text
先分类
    ↓
如果预测类别是“科技”
    ↓
优先在科技类新闻库中找相似新闻
    ↓
返回Top-5
```

如果同类别新闻不足，再回退到全库推荐：

```text
同类别新闻数量 >= top_k：在同类别中推荐
同类别新闻数量 < top_k：从全库补足
```

这个设计比直接全库推荐更合理。

------

# 十一、系统界面设计

总体效果为：接收用户输入新闻，并展示分类结果、关键词、自动摘要和相似新闻推荐结果

 Streamlit or qt待定

# 十二、实验设计

## 1. 分类实验(自由设计)

实验模型：

```text
TextCNN
BiLSTM
BiLSTM-Attention
Chinese-RoBERTa
```

评价指标：

```text
Accuracy
Precision
Recall
F1-score
Confusion Matrix
```

结果表：

| 模型             | Accuracy | Precision | Recall | F1   |
| ---------------- | -------- | --------- | ------ | ---- |
| TF-IDF + LR      |          |           |        |      |
| TextCNN          |          |           |        |      |
| BiLSTM           |          |           |        |      |
| BiLSTM-Attention |          |           |        |      |
| Chinese-RoBERTa  |          |           |        |      |

需要画：

```text
训练 Loss 曲线
验证 Accuracy 曲线
不同模型 Accuracy/F1 对比图
混淆矩阵
```

分析角度：

```text
TF-IDF + LR训练最快，但语义表达能力有限；
TextCNN能捕捉局部短语特征；
BiLSTM能建模上下文顺序；
Attention机制能突出关键特征；
Chinese-RoBERTa利用预训练语义知识，整体效果最好。
```

------

## 2. 自动摘要实验（成员B）

### 2.1 实验数据

```text
数据集：LCSTS 中文摘要数据集

训练集：用于微调 Pegasus-base
验证集：用于调参和选择模型
测试集：用于比较不同摘要方法
```

### 2.2 对比方法

```text
S1：TextRank
S2：Pegasus-base zero-shot
S3：Pegasus-base + LCSTS 微调
S4：Randeng-Pegasus-238M-Summary-Chinese
```

### 2.3 评价指标

```text
ROUGE-1
ROUGE-2
ROUGE-L
BERTScore
摘要长度
压缩率
```

### 2.4 结果表

| 方法 | ROUGE-1 | ROUGE-2 | ROUGE-L | BERTScore | 平均摘要长度 | 平均压缩率 |
| ---- | ------- | ------- | ------- | --------- | ------------ | ---------- |
| TextRank |  |  |  |  |  |  |
| Pegasus-base zero-shot |  |  |  |  |  |  |
| Pegasus-base + LCSTS 微调 |  |  |  |  |  |  |
| Randeng-Pegasus-238M-Summary-Chinese |  |  |  |  |  |  |

### 2.5 分析角度

```text
1. TextRank 是抽取式摘要，结果稳定但压缩和改写能力有限。
2. Pegasus-base zero-shot 可以体现未经过摘要微调时的生成效果。
3. Pegasus-base + LCSTS 微调用于体现成员 B 自己训练摘要模型的效果。
4. Randeng-Pegasus-238M-Summary-Chinese 作为公开中文摘要强基线。
5. ROUGE 和 BERTScore 共同评价字面重合与语义相似度，摘要长度和压缩率用于比较不同方法的压缩能力。
```

------

## 3. 关键词提取实验（成员B）

### 3.1 实验数据

```text
数据集：CSL 中文科学文献数据集

输入：论文标题 + 摘要
标准答案：人工关键词
```

### 3.2 对比方法

```text
K1：TF-IDF
K2：TextRank
K3：KeyBERT
```

### 3.3 评价指标

```text
Precision@5
Recall@5
F1@5
Precision@10
Recall@10
F1@10
```

### 3.4 结果表

| 方法 | P@5 | R@5 | F1@5 | P@10 | R@10 | F1@10 |
| ---- | --- | --- | ---- | ---- | ---- | ----- |
| TF-IDF |  |  |  |  |  |  |
| TextRank |  |  |  |  |  |  |
| KeyBERT |  |  |  |  |  |  |

### 3.5 分析角度

```text
1. TF-IDF 作为统计方法基线，速度快但语义能力有限。
2. TextRank 作为图排序方法，能够利用词语共现关系。
3. KeyBERT 利用预训练语义向量，理论上更能捕捉语义相关关键词。
4. P@K、R@K、F1@K 衡量关键词命中情况
```

------

## 4. 推荐实验（自由设计）


推荐模块不一定有标准标签，可以做案例分析和方法对比。

方法：

```text
TF-IDF + Cosine Similarity
Sentence-BERT + Cosine Similarity
```

展示表：

| 方法          | Top-1相似新闻 | Top-2相似新闻 | Top-3相似新闻 | 分析             |
| ------------- | ------------- | ------------- | ------------- | ---------------- |
| TF-IDF        | ……            | ……            | ……            | 关键词重合较明显 |
| Sentence-BERT | ……            | ……            | ……            | 语义相关性更强   |

还可以统计：

```text
Top-K推荐新闻中与输入新闻类别一致的比例
```

比如：

```text
输入科技新闻，Top-5中有4篇属于科技类
类别一致率 = 4 / 5 = 80%
```

这个指标简单，但能说明推荐模块合理。
