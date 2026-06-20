#将原始.txt转为train/val/test.csv
import os
import re
import csv
import json


train_file = "data/raw/cnews.train.txt"
val_file = "data/raw/cnews.val.txt"
test_file = "data/raw/cnews.test.txt"

out_dir = "data/processed"

train_csv = "data/processed/train.csv"
val_csv = "data/processed/val.csv"
test_csv = "data/processed/test.csv"
news_csv = "data/processed/news.csv"

label2id_file = "data/processed/label2id.json"
id2label_file = "data/processed/id2label.json"


def clean_text(text):
    text = re.sub(r"<.*?>", "", text)

    text = text.replace("\n", " ")
    text = text.replace("\r", " ")

    text = re.sub(r"\s+", " ", text)

    text = text.strip()

    return text


def read_cnews(file_name, split_name):
    data = []

    f = open(file_name, "r", encoding="utf-8")

    index = 1

    for line in f:
        line = line.strip()

        if line == "":
            continue

        # cnews 一般格式：
        # 类别 \t 新闻正文
        if "\t" in line:
            parts = line.split("\t", 1)
        else:
            parts = line.split(" ", 1)

        if len(parts) != 2:
            continue

        label = parts[0]
        content = parts[1]

        content = clean_text(content)

        if len(content) < 5:
            continue

        news_id = split_name + "_" + str(index).zfill(6)

        item = {
            "news_id": news_id,
            "content": content,
            "label": label
        }

        data.append(item)

        index += 1

    f.close()

    return data


def build_label_id(all_data):
    labels = []

    for item in all_data:
        label = item["label"]

        if label not in labels:
            labels.append(label)

    label2id = {}
    id2label = {}

    for i in range(len(labels)):
        label = labels[i]
        label2id[label] = i
        id2label[i] = label

    return label2id, id2label


def add_label_id(data, label2id):
    for item in data:
        label = item["label"]
        item["label_id"] = label2id[label]

    return data


def save_csv(data, save_file):
    f = open(save_file, "w", encoding="utf-8-sig", newline="")

    fieldnames = ["news_id", "content", "label", "label_id"]

    writer = csv.DictWriter(f, fieldnames=fieldnames)

    writer.writeheader()

    for item in data:
        writer.writerow(item)

    f.close()


def save_json(data, save_file):
    f = open(save_file, "w", encoding="utf-8")
    json.dump(data, f, ensure_ascii=False, indent=4)
    f.close()


def main():
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    train_data = read_cnews(train_file, "train")
    val_data = read_cnews(val_file, "val")
    test_data = read_cnews(test_file, "test")

    all_data = train_data + val_data + test_data

    label2id, id2label = build_label_id(all_data)

    train_data = add_label_id(train_data, label2id)
    val_data = add_label_id(val_data, label2id)
    test_data = add_label_id(test_data, label2id)
    all_data = add_label_id(all_data, label2id)

    save_csv(train_data, train_csv)
    save_csv(val_data, val_csv)
    save_csv(test_data, test_csv)
    save_csv(all_data, news_csv)

    save_json(label2id, label2id_file)
    save_json(id2label, id2label_file)

    print("数据处理完成")
    print("训练集数量：", len(train_data))
    print("验证集数量：", len(val_data))
    print("测试集数量：", len(test_data))
    print("总数据数量：", len(all_data))
    print("类别映射：", label2id)


if __name__ == "__main__":
    main()