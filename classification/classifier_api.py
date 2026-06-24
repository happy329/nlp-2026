# 1. C成员在系统中只需要调用 predict_class(text)
# 2. 直接运行本文件时，会进入命令行测试模式
# 3. 系统界面输入的新闻文本，由 pipeline.py 传给 predict_class，不会触发 input()
import argparse
import csv
import json
import logging
import os
import re
import sys
import warnings
from typing import Dict, List, Optional

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API.*")

import jieba
import torch
import torch.nn.functional as F

jieba.setLogLevel(logging.ERROR)


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import config as cfg
from classification.bilstm import BiLSTM
from classification.bilstm_attention import BiLSTMAttention
from classification.textcnn import TextCNN


VOCAB_FILE = os.path.join(PROJECT_DIR, cfg.vocab_file)
ID2LABEL_FILE = os.path.join(PROJECT_DIR, cfg.id2label_file)
STOPWORDS_FILE = os.path.join(PROJECT_DIR, cfg.stopwords_file)

MODEL_FILES = {
    "textcnn": os.path.join(PROJECT_DIR, cfg.textcnn_model_path),
    "bilstm": os.path.join(PROJECT_DIR, cfg.bilstm_model_path),
    "bilstm_attention": os.path.join(PROJECT_DIR, cfg.bilstm_attention_model_path),
}

DEFAULT_MODEL = "bilstm_attention"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab: Optional[Dict[str, int]] = None
file_id2label: Optional[Dict[int, str]] = None
stopwords: Optional[set] = None
loaded_models: Dict[str, Dict] = {}


def clean_text(text: str) -> str:
    text = str(text)
    text = re.sub(r"<.*?>", "", text)
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_vocab() -> Dict[str, int]:
    global vocab
    if vocab is None:
        with open(VOCAB_FILE, "r", encoding="utf-8") as f:
            vocab = json.load(f)
    return vocab


def load_file_id2label() -> Dict[int, str]:
    global file_id2label
    if file_id2label is None:
        with open(ID2LABEL_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        file_id2label = {int(k): v for k, v in data.items()}
    return file_id2label


def load_stopwords() -> set:
    global stopwords
    if stopwords is not None:
        return stopwords

    stopwords = set()
    if os.path.exists(STOPWORDS_FILE):
        with open(STOPWORDS_FILE, "r", encoding="utf-8") as f:
            stopwords = {line.strip() for line in f if line.strip()}
    return stopwords


def get_model_path(model_name: str) -> str:
    if model_name not in MODEL_FILES:
        raise ValueError("model_name 只能是 textcnn / bilstm / bilstm_attention")
    return MODEL_FILES[model_name]


def normalize_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        new_state_dict[new_key] = value
    return new_state_dict


def get_id2label(checkpoint: Dict) -> Dict[int, str]:
    label2id = checkpoint.get("label2id")
    if label2id:
        return {int(label_id): label for label, label_id in label2id.items()}
    return load_file_id2label()


def build_model(model_name: str, checkpoint: Dict) -> torch.nn.Module:
    state_dict = normalize_state_dict(checkpoint["model_state_dict"])
    pretrained_embedding = state_dict["embedding.weight"].detach().clone()

    num_classes = int(checkpoint.get("num_classes", len(get_id2label(checkpoint))))
    hidden_dim = int(checkpoint.get("hidden_dim", cfg.hidden_dim))
    pad_id = int(checkpoint.get("pad_id", 0))

    if model_name == "textcnn":
        model = TextCNN(
            pretrained_embedding=pretrained_embedding,
            num_classes=num_classes,
            pad_id=pad_id,
        )
    elif model_name == "bilstm":
        model = BiLSTM(
            pretrained_embedding=pretrained_embedding,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            pad_id=pad_id,
        )
    elif model_name == "bilstm_attention":
        model = BiLSTMAttention(
            pretrained_embedding=pretrained_embedding,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            pad_id=pad_id,
        )
    else:
        raise ValueError("model_name 只能是 textcnn / bilstm / bilstm_attention")

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def load_model(model_name: str = DEFAULT_MODEL) -> Dict:
    if model_name in loaded_models:
        return loaded_models[model_name]

    model_path = get_model_path(model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    checkpoint = torch.load(model_path, map_location=DEVICE)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"模型文件格式不正确: {model_path}")

    bundle = {
        "model": build_model(model_name, checkpoint),
        "id2label": get_id2label(checkpoint),
        "max_len": int(checkpoint.get("max_len", cfg.max_len)),
        "pad_id": int(checkpoint.get("pad_id", 0)),
    }
    loaded_models[model_name] = bundle
    return bundle


def text_to_ids(text: str, max_len: int, pad_id: int) -> List[int]:
    vocab_data = load_vocab()
    stopword_data = load_stopwords()

    words = jieba.lcut(clean_text(text))
    unk_id = vocab_data.get("<UNK>", 1)

    ids = []
    for word in words:
        word = word.strip()
        if not word or word in stopword_data:
            continue
        ids.append(vocab_data.get(word, unk_id))

    ids = ids[:max_len]
    if len(ids) < max_len:
        ids.extend([pad_id] * (max_len - len(ids)))
    return ids


def format_prediction(probs_tensor: torch.Tensor, id2label: Dict[int, str], top_k: int) -> Dict:
    pred_id = int(torch.argmax(probs_tensor).item())
    pred_label = id2label[pred_id]

    top_k = min(top_k, len(id2label))
    top_indices = torch.topk(probs_tensor, k=top_k).indices.tolist()
    top_labels = [
        {
            "label": id2label[int(i)],
            "label_id": int(i),
            "prob": round(float(probs_tensor[int(i)].item()), 6),
        }
        for i in top_indices
    ]

    return {
        "label": pred_label,
        "label_id": pred_id,
        "prob": round(float(probs_tensor[pred_id].item()), 6),
        "top_k": top_labels,
    }


def predict_class(text: str, model_name: str = DEFAULT_MODEL, top_k: int = 5) -> Dict:
    bundle = load_model(model_name)
    ids = text_to_ids(text, bundle["max_len"], bundle["pad_id"])
    x = torch.LongTensor([ids]).to(DEVICE)

    with torch.no_grad():
        logits = bundle["model"](x)
        if isinstance(logits, tuple):
            logits = logits[0]
        probs_tensor = F.softmax(logits, dim=1)[0].cpu()

    return format_prediction(probs_tensor, bundle["id2label"], top_k)


def predict_batch(texts: List[str], model_name: str = DEFAULT_MODEL, top_k: int = 5) -> List[Dict]:
    return [predict_class(text, model_name=model_name, top_k=top_k) for text in texts]


def print_result(result: Dict) -> None:
    print("=" * 60)
    print("预测类别:", result["label"])
    print("类别编号:", result["label_id"])
    print("置信度:", result["prob"])
    print("Top K:")
    for item in result["top_k"]:
        print(f"  {item['label_id']:>2}  {item['label']:<4}  {item['prob']:.6f}")


def read_multiline_text() -> Optional[str]:
    print("\n请粘贴新闻全文，输入 END 单独一行后开始预测；输入 q 或 quit 退出。")
    lines = []

    while True:
        try:
            line = input()
        except EOFError:
            if not lines:
                return None
            break

        command = line.strip().lower()
        if not lines and command in {"q", "quit", "exit"}:
            return None
        if line.strip() == "END":
            break

        lines.append(line)

    text = "\n".join(lines).strip()
    if not text:
        print("文本不能为空。")
        return ""
    return text


def run_interactive(model_name: str, top_k: int) -> None:
    print("中文新闻分类推理系统")
    print("支持多行新闻文本。每次输入完成后，用 END 单独一行提交预测。")
    print("当前模型:", model_name)

    while True:
        text = read_multiline_text()
        if text is None:
            break
        if not text:
            continue

        result = predict_class(text, model_name=model_name, top_k=top_k)
        print_result(result)


def predict_file(input_file: str, output_file: str, model_name: str, top_k: int) -> None:
    with open(input_file, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if "content" not in reader.fieldnames:
            raise ValueError("输入 CSV 必须包含 content 列")
        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    for row in rows:
        result = predict_class(row["content"], model_name=model_name, top_k=top_k)
        row["pred_label"] = result["label"]
        row["pred_label_id"] = result["label_id"]
        row["pred_prob"] = result["prob"]

    output_fields = fieldnames + ["pred_label", "pred_label_id", "pred_prob"]
    with open(output_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"批量预测完成，结果已保存到: {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="中文新闻分类推理系统")
    parser.add_argument("--text", type=str, help="单条新闻文本")
    parser.add_argument("--input_file", type=str, help="批量预测 CSV 文件，必须包含 content 列")
    parser.add_argument("--output_file", type=str, default="predict_result.csv", help="批量预测结果保存路径")
    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL,
        choices=["textcnn", "bilstm", "bilstm_attention"],
        help="选择用于推理的模型",
    )
    parser.add_argument("--top_k", type=int, default=5, help="展示概率最高的类别数量")
    parser.add_argument("--json", action="store_true", help="单条预测时用 JSON 格式输出")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.text:
        result = predict_class(args.text, model_name=args.model_name, top_k=args.top_k)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print_result(result)
        return

    if args.input_file:
        predict_file(args.input_file, args.output_file, args.model_name, args.top_k)
        return

    run_interactive(args.model_name, args.top_k)


if __name__ == "__main__":
    main()
