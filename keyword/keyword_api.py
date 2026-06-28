import argparse
import json
import os
import sys
from typing import Dict, List, Union


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)
KEYWORD_DIR = os.path.dirname(os.path.abspath(__file__))
if KEYWORD_DIR not in sys.path:
    sys.path.insert(0, KEYWORD_DIR)

import config as cfg
from keybert_keywords import extract_keybert_keywords
from textrank_keywords import extract_textrank_keywords
from tfidf_keywords import extract_tfidf_keywords


def extract_keywords(
    text: str,
    top_k: int = cfg.keyword_top_k,
    method: str = "tfidf",
) -> Union[List[str], Dict[str, List[str]]]:
    method = method.lower()

    if method == "tfidf":
        return extract_tfidf_keywords(text, top_k=top_k)
    if method == "textrank":
        return extract_textrank_keywords(text, top_k=top_k)
    if method == "keybert":
        return extract_keybert_keywords(text, top_k=top_k)
    if method in {"all", "both"}:
        result: Dict[str, List[str]] = {
            "tfidf": extract_tfidf_keywords(text, top_k=top_k),
            "textrank": extract_textrank_keywords(text, top_k=top_k),
        }
        try:
            result["keybert"] = extract_keybert_keywords(text, top_k=top_k)
        except Exception as exc:
            result["keybert"] = []
            result["keybert_error"] = [str(exc)]
        return result

    raise ValueError("method 只能是 tfidf / textrank / keybert / all")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="关键词提取统一接口")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--method", type=str, default="tfidf")
    parser.add_argument("--top_k", type=int, default=cfg.keyword_top_k)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(extract_keywords(args.text, top_k=args.top_k, method=args.method), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
