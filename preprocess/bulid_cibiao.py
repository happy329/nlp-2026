#构建词表
import csv
import os
import config as cfg

train_fenci_file = cfg.train_fenci_file
out_dir = cfg.processed_dir
vocab_file = cfg.vocab_txt_file


def read_words(file_name):
    word_count = {}

    f = open(file_name, "r", encoding="utf-8-sig")
    reader = csv.DictReader(f)

    for row in reader:
        content = row["fenci_content"]
        words = content.split(" ")

        for word in words:
            word = word.strip()

            if word == "":
                continue

            if word not in word_count:
                word_count[word] = 1
            else:
                word_count[word] += 1

    f.close()

    return word_count


def save_vocab(word_count, save_file):
    words = list(word_count.items())
    words = sorted(words, key=lambda x: x[1], reverse=True)

    f = open(save_file, "w", encoding="utf-8")

    f.write("<PAD>\n")
    f.write("<UNK>\n")

    for word, count in words:
        f.write(word + "\t" + str(count) + "\n")

    f.close()

    print("词表保存完成：", save_file)
    print("词表大小：", len(words) + 2)


def main():
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    word_count = read_words(train_fenci_file)
    save_vocab(word_count, vocab_file)


if __name__ == "__main__":
    main()
