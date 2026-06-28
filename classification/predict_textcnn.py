#测试验证脚本，最后没有用，后续改为_api.py接口
import json
import re
import jieba
import torch
import torch.nn.functional as F
import sys

sys.path.append(".")

from classification.textcnn import TextCNN


model_file = "saved_models/textcnn.pt"
vocab_file = "data/processed/vocab.json"
id2label_file = "data/processed/id2label.json"
stopwords_file = "data/processed/hit_stopwords.txt"


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


def text_to_ids(words, vocab, max_len):
    pad_id = vocab.get("<PAD>", 0)
    unk_id = vocab.get("<UNK>", 1)

    ids = []

    for word in words:
        ids.append(vocab.get(word, unk_id))

    if len(ids) > max_len:
        ids = ids[:max_len]

    if len(ids) < max_len:
        ids = ids + [pad_id] * (max_len - len(ids))

    return ids


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_model():
    device = get_device()

    checkpoint = torch.load(model_file, map_location=device)

    vocab_size = checkpoint["vocab_size"]
    embed_dim = checkpoint["embed_dim"]
    num_classes = checkpoint["num_classes"]
    pad_id = checkpoint["pad_id"]

    model = TextCNN(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_classes=num_classes,
        pad_id=pad_id
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, checkpoint, device


def predict_text(text):
    with open(vocab_file, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    with open(id2label_file, "r", encoding="utf-8") as f:
        id2label = json.load(f)

    stopwords = load_stopwords()

    model, checkpoint, device = load_model()

    max_len = checkpoint["max_len"]

    words = segment_text(text, stopwords)
    ids = text_to_ids(words, vocab, max_len)

    input_ids = torch.tensor([ids], dtype=torch.long)
    input_ids = input_ids.to(device)

    with torch.no_grad():
        outputs = model(input_ids)
        probs = F.softmax(outputs, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()

    pred_label = id2label[str(pred_id)]

    prob_dict = {}

    for i in range(probs.size(1)):
        label = id2label[str(i)]
        prob = probs[0][i].item()
        prob_dict[label] = round(prob, 4)

    result = {
        "label": pred_label,
        "label_id": pred_id,
        "probs": prob_dict,
        "cut_words": words[:30]
    }

    return result


if __name__ == "__main__":
    text = """
    近日，国内某科技公司发布新一代人工智能芯片，该芯片主要面向大模型训练和推理场景，
    能够提升计算效率，降低算力成本。业内人士认为，人工智能芯片产业将迎来新的发展机会。
    """

    result = predict_text(text)

    print("预测类别:", result["label"])
    print("类别编号:", result["label_id"])
    print("分类概率:")

    for label, prob in result["probs"].items():
        print(label, ":", prob)

    print("分词结果:", result["cut_words"])