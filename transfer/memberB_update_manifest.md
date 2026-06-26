# 成员 B 云端更新包清单

生成位置: `/root/autodl-tmp/nlp-2026/transfer/memberB_update_20260626.tar.gz`

## 打包原则

- 只包含本次云端相对仓库有改动、或成员 B 本地复现实验必需的文件。
- 包含最终可推理摘要模型，不包含训练 checkpoint 目录。
- 不包含 `__pycache__`、`.gitkeep`、临时缓存和未变化文件。

## 修改文件

- `.gitignore`
- `README.md`
- `classification/bilstm.py`
- `classification/bilstm_attention.py`
- `classification/classifier_api.py`
- `classification/predict_textcnn.py`
- `classification/textcnn.py`
- `config.py`
- `data/processed/id2label.json`
- `data/processed/label2id.json`
- `pipeline.py`
- `preprocess/jieba_fenci.py`
- `preprocess/pre.py`
- `preprocess/word2vec.py`
- `train/train_bilstm.py`
- `train/train_bilstm_attention.py`
- `train/train_textcnn.py`
- `visualization/plot_news_category.py`

## 新增源码与说明文件

- `common/__init__.py`
- `common/text_utils.py`
- `evaluate/eval_keywords.py`
- `evaluate/eval_summary.py`
- `keyword/__init__.py`
- `keyword/keybert_keywords.py`
- `keyword/keyword_api.py`
- `keyword/prepare_csl.py`
- `keyword/textrank_keywords.py`
- `keyword/tfidf_keywords.py`
- `summarization/model_utils.py`
- `summarization/pegasus_base_summary.py`
- `summarization/pegasus_finetuned_summary.py`
- `summarization/pegasus_summary.py`
- `summarization/prepare_lcsts.py`
- `summarization/randeng_pegasus_summary.py`
- `summarization/randeng_tokenizer.py`
- `summarization/summary_api.py`
- `summarization/textrank_summary.py`
- `summarization/train_summary_model.py`
- `第三题方案_成员B.md`

## 新增实验数据

- `data/LCSTS/train_small.csv`
- `data/LCSTS/valid_small.csv`
- `data/LCSTS/test_public_small.csv`
- `data/CSL/csl.csv`

说明: `data/LCSTS/` 和 `data/CSL/` 被忽略规则覆盖，Git 状态不会完整显示，但这些文件是本地复现成员 B 训练/评价所需的数据子集或整理文件。

## 新增模型文件

- `saved_models/summary_model/config.json`
- `saved_models/summary_model/generation_config.json`
- `saved_models/summary_model/model.safetensors`
- `saved_models/summary_model/special_tokens_map.json`
- `saved_models/summary_model/tokenizer_config.json`
- `saved_models/summary_model/training_args.bin`
- `saved_models/summary_model/vocab.txt`

说明: 这是最终摘要模型，可以直接用于推理和评价。

## 新增结果文件

- `results/rouge_score.csv`
- `results/bertscore.csv`
- `results/summary_length.csv`
- `results/summary_examples.csv`
- `results/keyword_scores.csv`
- `results/keyword_examples.csv`
- `results/figures/summary_compare.png`
- `results/figures/keyword_method_compare.png`
- `results/summary_train.log`
- `results/summary_eval.log`
- `results/keyword_eval.log`

## 删除文件

无。

## 未打包但云端存在的可选文件

- `saved_models/summary_model/checkpoint-8000/`
- `saved_models/summary_model/checkpoint-9375/`

说明: 这两个目录用于训练恢复，包含 `optimizer.pt` 等大文件，合计约 5GB 以上；本地如果只需要运行最终模型、查看结果和复现实验报告，不需要下载。
