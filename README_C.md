# 中文新闻分析系统

集新闻分类、关键词提取、自动摘要、相似新闻推荐于一体的中文 NLP 系统。

---

## 功能与现状

| 模块 | 状态 | 说明 |
|------|------|------|
| 关键词提取 | ✅ 完成 | jieba TF-IDF / TextRank，离线可用 |
| 抽取式摘要 | ✅ 完成 | 真实 TextRank（TF-IDF 建图 + PageRank），离线可用 |
| 相似推荐（TF-IDF）| ✅ 完成 | 离线可用 |
| 相似推荐（SBERT）| ✅ 完成 | 预构建索引，延迟加载，离线可用 |
| 新闻分类 | 🔧 待接入 | 接口就位，模型权重在 `fenlei_model.pt/`，需成员A加载 |
| 生成式摘要 | 🔧 待接入 | T5 / Pegasus 接口就位，需联网下载模型 |

---

## 项目结构

```
.
├── config.py                      # 全局配置（路径、参数）
├── pipeline.py                    # 统一调用入口
├── app.py                        # Streamlit Web 界面
├── requirements.txt              # 依赖列表
│
├── classification/               # 新闻分类（🔧 待成员A接入真实模型）
│   └── classifier_api.py
├── preprocess/
│   └── keyword.py                # 关键词提取（✅ 完成）
├── summarization/                # 摘要生成
│   └── summary_api.py           # ✅ TextRank离线可用；T5/Pegasus联网后可用
├── recommendation/
│   ├── build_index.py            # 构建离线索引
│   ├── tfidf_recommender.py     # ✅ 离线可用
│   ├── sbert_recommender.py     # ✅ 预构建索引，延迟加载
│   └── recommend_api.py         # 统一推荐 API
├── visualization/
│   ├── plot_embedding.py        # t-SNE / PCA 降维
│   ├── plot_category_distribution.py
│   └── plot_similarity.py
│
├── cnews_data/cnews_data/        # 原始新闻数据
├── vocab/vocab/                  # 词典与停用词
├── saved_index/                 # ✅ 已预构建索引
│   ├── tfidf_vectorizer.pkl / tfidf_matrix.pkl
│   ├── sbert_embeddings.npy / news_ids.json
├── fenlei_model.pt/             # 分类模型权重（🔧 待接入）
└── saved_models/summary_model/   # 摘要模型（联网后下载）
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

> **注意**：SBERT、T5、Pegasus 模型需要联网从 HuggingFace 下载。
> 如果不联网，**关键词、TextRank 摘要、TF-IDF 推荐**均可正常使用。

### 2. 构建索引（如需重建）

索引已预构建在 `saved_index/`，如需重新构建：

```bash
python recommendation/build_index.py
```

- `tfidf_vectorizer.pkl` / `tfidf_matrix.pkl` — TF-IDF 索引
- `sbert_embeddings.npy` — SBERT 向量（网络不畅时跳过）

### 3. 调用方式

#### 方式 A：Web 界面（推荐）

```bash
streamlit run app.py
```

浏览器打开 `http://localhost:8501`。

#### 方式 B：Python 代码

```python
from pipeline import run_pipeline

result = run_pipeline(
    "谷歌投资游戏平台Stickery获得融资，该平台教孩子基本数学知识",
    top_k=5,
    recommend_method="tfidf"   # 或 "sbert"
)
```

返回结构：

```python
{
    "classification": {"label": "游戏", "label_id": 8, "probs": {...}},
    "keywords": ["游戏", "移动", "平台", "融资", ...],
    "summary": {
        "t5_summary": "...",          # 联网后可用
        "pegasus_summary": "...",      # 联网后可用
        "textrank_summary": "..."      # 离线可用 ✅
    },
    "recommendations": [
        {"title": "...", "label": "游戏", "score": 0.4521, ...},
        ...
    ]
}
```

#### 方式 C：单独调用各模块

```python
# 关键词
from preprocess.keyword import extract_keywords
extract_keywords(text, top_k=10, method="tfidf")

# TextRank 摘要（离线）
from summarization.summary_api import SummaryAPI
SummaryAPI().textrank_summary(text, top_k=3)

# TF-IDF 推荐
from recommendation.tfidf_recommender import TFIDFRecommender
TFIDFRecommender().load().recommend(text, top_k=5)
```

### 4. 可视化

```python
from visualization.plot_category_distribution import plot_category_distribution
plot_category_distribution()

from visualization.plot_embedding import plot_embedding_visualization
plot_embedding_visualization(method='tsne')   # t-SNE
plot_embedding_visualization(method='pca')    # PCA
```

---

## 成员任务说明

### 成员A — 新闻分类

接口已定义，模型权重在 `fenlei_model.pt/`（包含 `textcnn.pt`、`bilstm.pt`、`bilstm_attention.pt`）。

在 `classification/classifier_api.py` 的 `predict_class` 中加载 `.pt` 文件进行推理即可。

### 成员B — 生成式摘要

接口已定义，模型 ID：

- **T5**：`twwch/mt5-base-summary`（多语言 T5，专为中文摘要微调）
- **Pegasus**：`IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese-V1`

联网后自动下载。在 `summarization/summary_api.py` 的 `summarize_with_t5` / `summarize_with_pegasus` 中接入真实推理即可。

---

## 常见问题

**Q: 离线可以用哪些功能？**

> 关键词提取、TextRank 抽取摘要、TF-IDF 推荐，全部可用，无需联网。

**Q: 如何启用 SBERT 推荐？**
> `recommend_method="sbert"`，首次运行会联网下载模型（~500MB），下载后离线可用。

**Q: 如何启用 T5/Pegasus 生成式摘要？**
> 联网状态下首次调用自动下载模型（T5 ~1.2GB，Pegasus ~1GB），下载后离线可用。

**Q: 分类结果是随机的？**
> 是的，`fenlei_model.pt/` 中的模型权重尚未加载，请联系成员A接入。

**Q: 如何重新构建索引？**
> `python recommendation/build_index.py`，已有索引会被覆盖。

**Q: Windows 上 HuggingFace 下载报 symlink 警告？**
> 可忽略，不影响功能。也可以开启开发者模式消除警告。
