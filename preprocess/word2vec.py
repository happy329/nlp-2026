import os
import json
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
import config as cfg

train_file = cfg.train_fenci_file 
vocab_file = cfg.vocab_file

out_dir = cfg.processed_dir
word2vec_model_file = cfg.word2vec_model_file
embedding_matrix_file = cfg.embedding_matrix_file

vector_size = cfg.vector_size
window = cfg.window
min_count = cfg.min_count


def load_sentences():
    df = pd.read_csv(train_file)

    sentences = []

    for i in range(len(df)):
        cut_text = str(df.iloc[i]["cut_text"])
        words = cut_text.split()

        if len(words) > 0:
            sentences.append(words)

    return sentences


def build_embedding_matrix(model, vocab):
    vocab_size = len(vocab)

    embedding_matrix = np.random.normal(
        loc=0,
        scale=0.1,
        size=(vocab_size, vector_size)
    )

    pad_id = vocab.get("<PAD>", 0)

    embedding_matrix[pad_id] = np.zeros(vector_size)

    for word, word_id in vocab.items():
        if word in model.wv:
            embedding_matrix[word_id] = model.wv[word]

    return embedding_matrix


def main():
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print("读取分词数据...")
    sentences = load_sentences()

    print("句子数量:", len(sentences))

    print("开始训练Word2Vec...")

    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=0,
        workers=4
    )

    model.save(word2vec_model_file)

    print("Word2Vec模型保存到:", word2vec_model_file)

    with open(vocab_file, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    embedding_matrix = build_embedding_matrix(model, vocab)

    np.save(embedding_matrix_file, embedding_matrix)

    print("Embedding矩阵保存到:", embedding_matrix_file)
    print("Embedding矩阵形状:", embedding_matrix.shape)


if __name__ == "__main__":
    main()