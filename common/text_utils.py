import csv
import json
import logging
import os
import re
import sys
from typing import Dict, Iterable, List, Optional, Sequence


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import config as cfg

try:
    import jieba

    jieba.setLogLevel(logging.ERROR)
except ImportError:  # pragma: no cover - handled at runtime for lightweight checks
    jieba = None


PUNCT_RE = re.compile(r"^[\W_]+$", re.UNICODE)
CHINESE_RE = re.compile(r"^[\u4e00-\u9fff]+$")


def project_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(PROJECT_DIR, path)


def ensure_dir(path: str) -> None:
    os.makedirs(project_path(path), exist_ok=True)


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(project_path(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def clean_text(text: str) -> str:
    text = "" if text is None else str(text)
    text = re.sub(r"<.*?>", "", text)
    text = text.replace("\u3000", " ")
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_sentences(text: str) -> List[str]:
    text = clean_text(text)
    if not text:
        return []

    pieces = re.split(r"([。！？!?；;])", text)
    sentences = []
    current = ""

    for piece in pieces:
        if not piece:
            continue
        current += piece
        if re.fullmatch(r"[。！？!?；;]", piece):
            sentence = current.strip()
            if sentence:
                sentences.append(sentence)
            current = ""

    if current.strip():
        sentences.append(current.strip())

    if len(sentences) <= 1 and len(text) > 120:
        sentences = [part.strip() for part in re.split(r"[，,]", text) if part.strip()]

    return sentences or [text]


def load_stopwords(path: Optional[str] = None) -> set:
    stopwords = set()
    stopword_path = project_path(path or cfg.stopwords_file)
    if os.path.exists(stopword_path):
        with open(stopword_path, "r", encoding="utf-8") as f:
            stopwords = {line.strip() for line in f if line.strip()}

    stopwords.update({" ", "\n", "\t", "的", "了", "和", "与", "及", "在", "是"})
    return stopwords


def fallback_cut(text: str, keep_single_char: bool = False) -> List[str]:
    tokens = []
    chunks = re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9]+", clean_text(text))
    for chunk in chunks:
        if CHINESE_RE.match(chunk):
            if keep_single_char or len(chunk) <= 2:
                tokens.extend(list(chunk) if keep_single_char else [chunk])
            else:
                tokens.extend(chunk[index : index + 2] for index in range(len(chunk) - 1))
        else:
            tokens.append(chunk.lower())
    return tokens


def tokenize(text: str, stopwords: Optional[set] = None, keep_single_char: bool = False) -> List[str]:
    stopwords = stopwords if stopwords is not None else load_stopwords()
    raw_words = fallback_cut(text, keep_single_char=keep_single_char) if jieba is None else jieba.lcut(clean_text(text))
    words = []
    for word in raw_words:
        word = word.strip()
        if not word or word in stopwords:
            continue
        if PUNCT_RE.match(word):
            continue
        if not keep_single_char and len(word) <= 1:
            continue
        words.append(word)
    return words


def tokenized_text(text: str, stopwords: Optional[set] = None) -> str:
    return " ".join(tokenize(text, stopwords=stopwords))


def normalize_keyword(word: str) -> str:
    word = clean_text(word).lower()
    word = re.sub(r"[\s_、,，;；|/\\:：.。!！?？\-—()\[\]{}（）【】《》<>]+", "", word)
    return word


def parse_keywords(value: str) -> List[str]:
    if value is None:
        return []

    text = clean_text(value)
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [clean_text(item) for item in data if clean_text(item)]
        except json.JSONDecodeError:
            pass

    parts = re.split(r"[_;；,，|/]", text)
    return [part.strip() for part in parts if part.strip()]


def unique_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        key = normalize_keyword(item)
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def read_csv_rows(path: str) -> List[Dict[str, str]]:
    with open(project_path(path), "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: str, rows: Sequence[Dict], fieldnames: Sequence[str]) -> None:
    ensure_parent(path)
    with open(project_path(path), "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
