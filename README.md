# nlp-2026
基于Transformer的中文新闻分类与摘要系统

## 1.数据集和分类模型权重获取

百度网盘链接：[点击下载](https://pan.baidu.com/s/1AfstxTgXyx23x8hm94sfGg?pwd=m3ah)
提取码：m3ah

里面有cnews_data.zip，包含train_csv, val_csv, test_csv， news_csv,解压后放到`data/processed/`

vocab.zip解压后放到`data/processed/`
fenlei_pt文件夹里有训练好的三种分类模型的.pt，放到`saved_models/`

成员 B 的 LCSTS 默认使用小规模子集，避免全量数据过大。项目中保留：

`data/LCSTS/train_small.csv`、`valid_small.csv`、`test_public_small.csv`。

如需重新生成小数据，请先把 LCSTS 原始 `jsonl` 文件放回 `data/LCSTS/raw/`，再运行 `python summarization/prepare_lcsts.py`。


## 2.文件目录说明

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
│		└── LCSTS/
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
├── common/
│   └── text_utils.py
│
├── summarization/
│   ├── prepare_lcsts.py
│   ├── train_summary_model.py
│   ├── textrank_summary.py
│   ├── pegasus_base_summary.py
│   ├── pegasus_finetuned_summary.py
│   ├── randeng_pegasus_summary.py
│   ├── pegasus_summary.py
│   └── summary_api.py
│
├── keyword/
│   ├── prepare_csl.py
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
│   ├── recommend_examples.csv
│   ├── rouge_score.csv
│   └── figures/
│       ├── category_distribution.png
│       ├── model_accuracy.png
│       ├── model_f1.png
│       ├── loss_curve.png
│       ├── confusion_matrix.png
│       ├── summary_compare.png
│       ├── similarity_topk.png
│       └── embedding_visualization.png
│
├── saved_models/
│   ├── textcnn.pt
│   ├── bilstm.pt
│   ├── bilstm_attention.pt
│   └── saved_models/summary_model/
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

