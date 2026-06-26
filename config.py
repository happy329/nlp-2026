# 数据目录
data_dir = "data"
raw_dir = "data/raw"
processed_dir = "data/processed"


# 原始.txt的路径
raw_train_file = "data/raw/cnews.train.txt"
raw_val_file = "data/raw/cnews.val.txt"
raw_test_file = "data/raw/cnews.test.txt"


# train/val/test/news.csv的路径
train_file = "data/processed/train.csv"
val_file = "data/processed/val.csv"
test_file = "data/processed/test.csv"
news_file = "data/processed/news.csv"

#train/val/test/news_fenci.csv的路径
train_fenci_file = "data/processed/train_fenci.csv"
val_fenci_file = "data/processed/val_fenci.csv"
test_fenci_file = "data/processed/test_fenci.csv"
news_fenci_file = "data/processed/news_fenci.csv"


# label2id/id2label的路径
label2id_file = "data/processed/label2id.json"
id2label_file = "data/processed/id2label.json"

# vocab的路径
vocab_file = "data/processed/vocab.json"
vocab_txt_file = "data/processed/vocab.txt"

# 停用词的路径
stopwords_file = "data/processed/hit_stopwords.txt"


# Word2Vec的路径
word2vec_model_file = "data/processed/word2vec.model"
embedding_matrix_file = "data/processed/embedding_matrix.npy"

word2vec_dim = 100
word2vec_window = 5
word2vec_min_count = 2
epochs_word2vec = 10

# 分类模型参数
max_len = 200#每条新闻分词后保留多少个词
batch_size = 64
epochs = 20
learning_rate = 0.001

num_classes = 13
hidden_dim = 128

#jieba分词
min_word_freq = 3

#word2vec
vector_size = 200#维度
window = 5
min_count = 2

# 模型保存路径
textcnn_model_path = "saved_models/textcnn.pt"
bilstm_model_path = "saved_models/bilstm.pt"
bilstm_attention_model_path = "saved_models/bilstm_attention.pt"



# 实验结果路径
metrics_file = "results/classification_metrics.csv"

textcnn_loss_fig = "results/figures/textcnn_loss_curve.png"
bilstm_loss_fig = "results/figures/bilstm_loss_curve.png"
bilstm_attention_loss_fig = "results/figures/bilstm_attention_loss_curve.png"

textcnn_cm_fig = "results/figures/textcnn_confusion_matrix.png"
bilstm_cm_fig = "results/figures/bilstm_confusion_matrix.png"
bilstm_attention_cm_fig = "results/figures/bilstm_attention_confusion_matrix.png"

model_acc_fig = "results/figures/model_accuracy.png"
model_f1_fig = "results/figures/model_f1.png"


# 推荐模块后面会用
sbert_index_file = "saved_index/sbert_embeddings.npy"
tfidf_matrix_file = "saved_index/tfidf_matrix.pkl"
tfidf_vectorizer_file = "saved_index/tfidf_vectorizer.pkl"
news_ids_file = "saved_index/news_ids.json"


# 成员B：自动摘要数据与模型路径
lcsts_dir = "data/LCSTS"
lcsts_raw_dir = "data/LCSTS/raw"
lcsts_train_raw_file = "data/LCSTS/raw/train.jsonl"
lcsts_val_raw_file = "data/LCSTS/raw/valid.jsonl"
lcsts_test_raw_file = "data/LCSTS/raw/test_public.jsonl"

# 全量整理文件，保留给需要完整实验时使用
lcsts_train_full_file = "data/LCSTS/train.csv"
lcsts_val_full_file = "data/LCSTS/valid.csv"
lcsts_test_full_file = "data/LCSTS/test_public.csv"

# 课设默认使用小规模子集，训练和评价速度更合适
lcsts_train_file = "data/LCSTS/train_small.csv"
lcsts_val_file = "data/LCSTS/valid_small.csv"
lcsts_test_file = "data/LCSTS/test_public_small.csv"
lcsts_eval_file = lcsts_val_file
lcsts_train_limit = 50000
lcsts_val_limit = 2000
lcsts_test_limit = 725

summary_model_dir = "saved_models/summary_model"
pegasus_base_model = "IDEA-CCNL/Randeng-Pegasus-238M-Chinese"
public_summary_model = "IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese"

summary_max_source_len = 512
summary_max_target_len = 64
summary_min_target_len = 10
summary_num_beams = 4
summary_train_epochs = 3
summary_train_batch_size = 4
summary_eval_batch_size = 4
summary_learning_rate = 5e-5

rouge_score_file = "results/rouge_score.csv"
bertscore_file = "results/bertscore.csv"
summary_length_file = "results/summary_length.csv"
summary_examples_file = "results/summary_examples.csv"
summary_compare_fig = "results/figures/summary_compare.png"


# 成员B：关键词提取数据路径与结果路径
csl_dir = "data/CSL"
csl_raw_dir = "data/CSL/raw"
csl_raw_file = "data/CSL/raw/csl_40k.tsv"
csl_file = "data/CSL/csl.csv"

keyword_scores_file = "results/keyword_scores.csv"
keyword_examples_file = "results/keyword_examples.csv"
keyword_compare_fig = "results/figures/keyword_method_compare.png"

keyword_top_k = 10
keyword_window_size = 4
keyword_max_features = 50000
keybert_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
