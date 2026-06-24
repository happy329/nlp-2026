#jieba分词
import re
import jieba
import os
import csv
import json
from collections import Counter
import sys
sys.path.append(".")
import config as cfg

train_file = cfg.train_file
val_file = cfg.val_file 
test_file = cfg.test_file

stopwords_file = cfg.stopwords_file

out_dir = cfg.processed_dir

train_out = cfg.train_fenci_file
val_out = cfg.val_fenci_file
test_out = cfg.test_fenci_file

vocab_out = cfg.vocab_file
vocab_txt_out = cfg.vocab_txt_file 

label2id_out = cfg.label2id_file
id2label_out = cfg.id2label_file

min_word_freq = cfg.min_word_freq #需要改


def simple_clean(text):
    text = str(text)

    text = re.sub(r"<.*?>", "", text)

    text = text.replace("\n", " ")
    text = text.replace("\r", " ")

    text = re.sub(r"\s+", " ", text)

    text = text.strip()

    return text


def load_stopwords():
    stopwords = set()

    f = open(stopwords_file, "r", encoding="utf-8")

    for line in f:
        word = line.strip()

        if word != "":
            stopwords.add(word)

    f.close()

    return stopwords


def segment_text(text, stopwords):
    text = simple_clean(text)

    words = jieba.lcut(text)

    new_words = []

    for word in words:
        word = word.strip()

        if word == "":
            continue

        if word in stopwords:
            continue

        new_words.append(word)

    return new_words


def read_data(read_file, stopwords):
    f = open(read_file, "r", encoding="utf-8-sig")

    reader = csv.DictReader(f)

    data = []
    all_words = []
    label2id = {}

    for row in reader:
        content = row["content"]
        words = segment_text(content, stopwords)

        if len(words) == 0:
            continue

        label = row["label"]
        label_id = row["label_id"]

        label2id[label] = int(label_id)

        item = {
            "news_id": row["news_id"],
            "words": words,
            "label": label,
            "label_id": label_id
        }

        data.append(item)

        for word in words:
            all_words.append(word)

    f.close()

    return data, all_words, label2id


def build_vocab(all_words):
    word_count = Counter(all_words)

    vocab = {}

    vocab["<PAD>"] = 0
    vocab["<UNK>"] = 1

    index = 2

    for word, count in word_count.most_common():
        if count >= min_word_freq:
            vocab[word] = index
            index += 1

    return vocab


def save_fenci_csv(data, save_file):
    out = open(save_file, "w", encoding="utf-8-sig", newline="")

    fieldnames = [
        "news_id",
        "cut_text",
        "label",
        "label_id"
    ]

    writer = csv.DictWriter(out, fieldnames=fieldnames)
    writer.writeheader()

    count = 0

    for item in data:
        row = {
            "news_id": item["news_id"],
            "cut_text": " ".join(item["words"]),
            "label": item["label"],
            "label_id": item["label_id"]
        }

        writer.writerow(row)
        count += 1

    out.close()

    print(save_file, "保存完成，数量：", count)


def save_json(obj, save_file):
    f = open(save_file, "w", encoding="utf-8")
    json.dump(obj, f, ensure_ascii=False, indent=4)
    f.close()

    print(save_file, "保存完成")


def save_vocab_txt(vocab, save_file):
    f = open(save_file, "w", encoding="utf-8")

    for word, word_id in vocab.items():
        f.write(word + "\t" + str(word_id) + "\n")

    f.close()

    print(save_file, "保存完成")


def build_id2label(label2id):
    id2label = {}

    for label, label_id in label2id.items():
        id2label[str(label_id)] = label

    return id2label


def main():
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    stopwords = load_stopwords()

    print("开始处理 train.csv")
    train_data, train_words, label2id = read_data(train_file, stopwords)

    print("开始处理 val.csv")
    val_data, val_words, val_label2id = read_data(val_file, stopwords)

    print("开始处理 test.csv")
    test_data, test_words, test_label2id = read_data(test_file, stopwords)

    vocab = build_vocab(train_words)

    id2label = build_id2label(label2id)

    save_fenci_csv(train_data, train_out)
    save_fenci_csv(val_data, val_out)
    save_fenci_csv(test_data, test_out)

    save_json(vocab, vocab_out)
    save_vocab_txt(vocab, vocab_txt_out)

    save_json(label2id, label2id_out)
    save_json(id2label, id2label_out)

    print("分词全部完成")
    print("训练集数量：", len(train_data))
    print("验证集数量：", len(val_data))
    print("测试集数量：", len(test_data))
    print("词表大小：", len(vocab))
    print("类别数量：", len(label2id))


if __name__ == "__main__":
    main()