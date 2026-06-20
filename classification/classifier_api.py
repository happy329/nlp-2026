#最后连入系统的接口
import json
import re
import jieba
import torch
import torch.nn.functional as F
import sys

sys.path.append(".")

from classification.textcnn import TextCNN
from classification.bilstm import BiLSTM
from classification.bilstm_attention import BiLSTMAttention


vocab_file = "data/processed/vocab.json"
id2label_file = "data/processed/id2label.json"
stopwords_file = "data/processed/hit_stopwords.txt"

textcnn_model_file = "saved_models/textcnn.pt"
bilstm_model_file = "saved_models/bilstm.pt"
bilstm_attention_model_file = "saved_models/bilstm_attention.pt"


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
        word_id = vocab.get(word, unk_id)
        ids.append(word_id)

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


def load_textcnn_model(device):
    checkpoint = torch.load(textcnn_model_file, map_location=device)

    model = TextCNN(
        vocab_size=checkpoint["vocab_size"],
        embed_dim=checkpoint["embed_dim"],
        num_classes=checkpoint["num_classes"],
        pad_id=checkpoint["pad_id"]
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, checkpoint


def load_bilstm_model(device):
    checkpoint = torch.load(bilstm_model_file, map_location=device)

    model = BiLSTM(
        vocab_size=checkpoint["vocab_size"],
        embed_dim=checkpoint["embed_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_classes=checkpoint["num_classes"],
        pad_id=checkpoint["pad_id"]
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, checkpoint


def load_bilstm_attention_model(device):
    checkpoint = torch.load(bilstm_attention_model_file, map_location=device)

    model = BiLSTMAttention(
        vocab_size=checkpoint["vocab_size"],
        embed_dim=checkpoint["embed_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_classes=checkpoint["num_classes"],
        pad_id=checkpoint["pad_id"]
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model, checkpoint


def predict_with_model(text, model, checkpoint, device, vocab, id2label):
    stopwords = load_stopwords()

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
        "probs": prob_dict
    }

    return result


def predict_class(text, model_name="textcnn"):
    with open(vocab_file, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    with open(id2label_file, "r", encoding="utf-8") as f:
        id2label = json.load(f)

    device = get_device()

    if model_name == "textcnn":
        model, checkpoint = load_textcnn_model(device)
        return predict_with_model(text, model, checkpoint, device, vocab, id2label)

    if model_name == "bilstm":
        model, checkpoint = load_bilstm_model(device)
        return predict_with_model(text, model, checkpoint, device, vocab, id2label)

    if model_name == "bilstm_attention":
        model, checkpoint = load_bilstm_attention_model(device)
        return predict_with_model(text, model, checkpoint, device, vocab, id2label)

    print("暂时不支持这个模型:", model_name)
    return None


if __name__ == "__main__":
    text = """
    近日，国内某科技公司发布新一代人工智能芯片，该芯片主要面向大模型训练和推理场景。
    业内人士认为，人工智能芯片产业将迎来新的发展机会。
    """

    model_names = ["textcnn", "bilstm", "bilstm_attention"]

    for model_name in model_names:
        print("=" * 50)
        print("模型:", model_name)

        result = predict_class(text, model_name=model_name)

        print("预测类别:", result["label"])
        print("类别编号:", result["label_id"])
        print("分类概率:")

        for label, prob in result["probs"].items():
            print(label, ":", prob)