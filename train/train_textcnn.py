import os
import sys
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import config as cfg

sys.path.append(".")

from classification.textcnn import TextCNN


train_file = cfg.train_fenci_file
val_file = cfg.val_fenci_file
test_file = cfg.test_fenci_file
vocab_file = cfg.vocab_file
label2id_file = cfg.label2id_file

model_save_path = cfg.textcnn_model_path
metrics_save_path = cfg.metrics_file
loss_fig_path = cfg.textcnn_loss_fig
cm_fig_path = cfg.textcnn_cm_fig

embedding_matrix_file = cfg.embedding_matrix_file

max_len = cfg.max_len #200
batch_size = cfg.batch_size #64
# embed_dim = 64 #128 这里不使用随机初始化，使用word2vec
epochs = cfg.epochs #
learning_rate = cfg.learning_rate


class NewsDataset(Dataset):
    def __init__(self, csv_file, vocab, max_len):
        self.data = pd.read_csv(csv_file)
        self.vocab = vocab
        self.max_len = max_len

        self.pad_id = vocab.get("<PAD>", 0)
        self.unk_id = vocab.get("<UNK>", 1)

    def __len__(self):
        return len(self.data)

    def text_to_ids(self, cut_text):
        words = str(cut_text).split()

        ids = []
        for word in words:
            ids.append(self.vocab.get(word, self.unk_id))

        if len(ids) > self.max_len:
            ids = ids[:self.max_len]

        if len(ids) < self.max_len:
            ids = ids + [self.pad_id] * (self.max_len - len(ids))

        return ids

    def __getitem__(self, index):
        row = self.data.iloc[index]

        cut_text = row["cut_text"]
        label_id = int(row["label_id"])

        ids = self.text_to_ids(cut_text)

        ids = torch.tensor(ids, dtype=torch.long)
        label_id = torch.tensor(label_id, dtype=torch.long)

        return ids, label_id


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()

    total_loss = 0
    all_preds = []
    all_labels = []

    for batch_index, (batch_x, batch_y) in enumerate(dataloader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(batch_y.cpu().numpy().tolist())

        total_loss += loss.item()

        if (batch_index + 1) % 100 == 0:
            print("已经训练 batch:", batch_index + 1, "/", len(dataloader),
                  "当前loss:", round(loss.item(), 4))

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)

    return avg_loss, acc

def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            preds = torch.argmax(outputs, dim=1)

            total_loss += loss.item()

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(batch_y.cpu().numpy().tolist())

    avg_loss = total_loss / len(dataloader)

    acc = accuracy_score(all_labels, all_preds)

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels,
        all_preds,
        average="macro",
        zero_division=0
    )

    return avg_loss, acc, precision, recall, f1, all_labels, all_preds


def save_metrics(model_name, acc, precision, recall, f1):
    os.makedirs("results", exist_ok=True)

    new_row = {
        "model": model_name,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

    if os.path.exists(metrics_save_path):
        df = pd.read_csv(metrics_save_path)

        df = df[df["model"] != model_name]
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])

    df.to_csv(metrics_save_path, index=False, encoding="utf-8-sig")


def plot_loss(train_losses, val_losses):
    os.makedirs("results/figures", exist_ok=True)

    plt.figure()
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker="o", label="train_loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, marker="o", label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("TextCNN Loss Curve")
    plt.legend()
    plt.savefig(loss_fig_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(labels, preds, label_names):
    os.makedirs("results/figures", exist_ok=True)

    cm = confusion_matrix(labels, preds)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm)
    plt.title("TextCNN Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar()

    tick_marks = range(len(label_names))
    plt.xticks(tick_marks, label_names, rotation=45)
    plt.yticks(tick_marks, label_names)

    for i in range(len(cm)):
        for j in range(len(cm[i])):
            plt.text(j, i, str(cm[i][j]), ha="center", va="center")

    plt.savefig(cm_fig_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    with open(vocab_file, "r", encoding="utf-8") as f:
        vocab = json.load(f)

    with open(label2id_file, "r", encoding="utf-8") as f:
        label2id = json.load(f)

    label_names = []
    for label, idx in sorted(label2id.items(), key=lambda x: x[1]):
        label_names.append(label)

    vocab_size = len(vocab)
    num_classes = len(label2id)
    pad_id = vocab.get("<PAD>", 0)

    print("词表大小:", vocab_size)
    print("类别数量:", num_classes)
    print("类别列表:", label_names)

    train_dataset = NewsDataset(train_file, vocab, max_len)
    val_dataset = NewsDataset(val_file, vocab, max_len)
    test_dataset = NewsDataset(test_file, vocab, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = get_device()
    print("使用设备:", device)

    embedding_matrix = np.load(embedding_matrix_file)
    embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float)

    model = TextCNN(
        pretrained_embedding=embedding_matrix,
        num_classes=num_classes,
        pad_id=pad_id
    )

    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    best_val_acc = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        val_loss, val_acc, val_precision, val_recall, val_f1, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print("=" * 50)
        print("Epoch:", epoch + 1)
        print("Train Loss:", round(train_loss, 4))
        print("Train Acc:", round(train_acc, 4))
        print("Val Loss:", round(val_loss, 4))
        print("Val Acc:", round(val_acc, 4))
        print("Val Precision:", round(val_precision, 4))
        print("Val Recall:", round(val_recall, 4))
        print("Val F1:", round(val_f1, 4))

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab_size": vocab_size,
                "embed_dim": embedding_matrix.shape[1],
                "num_classes": num_classes,
                "pad_id": pad_id,
                "max_len": max_len,
                "label2id": label2id,
                "use_word2vec": True
                }, model_save_path)

            print("保存当前最优模型:", model_save_path)

    plot_loss(train_losses, val_losses)

    print("=" * 50)
    print("开始测试集评估")

    checkpoint = torch.load(model_save_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, test_precision, test_recall, test_f1, test_labels, test_preds = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    print("Test Loss:", round(test_loss, 4))
    print("Test Acc:", round(test_acc, 4))
    print("Test Precision:", round(test_precision, 4))
    print("Test Recall:", round(test_recall, 4))
    print("Test F1:", round(test_f1, 4))

    print()
    print("分类报告:")
    print(classification_report(
        test_labels,
        test_preds,
        target_names=label_names,
        zero_division=0
    ))

    save_metrics("TextCNN", test_acc, test_precision, test_recall, test_f1)
    plot_confusion_matrix(test_labels, test_preds, label_names)

    print("=" * 50)
    print("TextCNN训练完成")
    print("模型保存到:", model_save_path)
    print("指标保存到:", metrics_save_path)
    print("Loss曲线保存到:", loss_fig_path)
    print("混淆矩阵保存到:", cm_fig_path)


if __name__ == "__main__":
    main()