import argparse
import os
import sys
from typing import Iterable, List


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from summarization.pegasus_summary import get_summarizer


def randeng_pegasus_summarize(text: str) -> str:
    return get_summarizer("randeng").summarize(text)


def randeng_pegasus_summarize_batch(texts: Iterable[str], batch_size: int = 4) -> List[str]:
    return get_summarizer("randeng").summarize_batch(texts, batch_size=batch_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Randeng-Pegasus-238M-Summary-Chinese 中文摘要")
    parser.add_argument("--text", type=str, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(randeng_pegasus_summarize(args.text))


if __name__ == "__main__":
    main()
