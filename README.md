# nlp-2026
基于Transformer的中文新闻分类与摘要系统

## 1.数据集获取

百度网盘链接：[点击下载](https://pan.baidu.com/s/17YnbinvVVT3_Jc6l15dhrQ?pwd=ppr4)

提取码：ppr4

里面包含cnews_raw，解压后放到`data/raw/`

cnews_csv包含train_csv, val_csv, test_csv， news_csv，解压后放到`data/processed/`

fenci_cibiao解压后放到`data/processed/`



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
├── summarization/
│		summarization/prepare_lcsts.py
│   summarization/train_summary_model.py
│   summarization/t5_summary.py 和(或) pegasus_summary.py
│   summarization/textrank_summary.py
│   summarization/summary_api.py
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

