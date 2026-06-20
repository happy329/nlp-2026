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

word2vec_dim = 200
word2vec_window = 5
word2vec_min_count = 2


# 分类模型参数
max_len = 200
batch_size = 32
epochs = 2
learning_rate = 0.001

num_classes = 10
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

