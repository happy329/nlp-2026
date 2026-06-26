import argparse
import json
import os
import sys
from typing import Dict


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from summarization.pegasus_base_summary import pegasus_base_summarize
from summarization.pegasus_finetuned_summary import pegasus_finetuned_summarize
from summarization.randeng_pegasus_summary import randeng_pegasus_summarize
from summarization.textrank_summary import textrank_summarize
from common.text_utils import clean_text


def summarize(text: str, method: str = "both", model_name: str = "randeng") -> Dict[str, str]:
    text = clean_text(text)
    method = method.lower()
    result: Dict[str, str] = {}

    if method in {"textrank", "both", "all"}:
        result["textrank_summary"] = textrank_summarize(text)

    if method in {"pegasus_base", "base", "zero_shot", "all"}:
        try:
            result["pegasus_base_summary"] = pegasus_base_summarize(text)
        except Exception as exc:
            result["pegasus_base_summary"] = ""
            result["pegasus_base_error"] = str(exc)

    if method in {"pegasus_finetuned", "finetuned", "local", "all"}:
        try:
            result["pegasus_finetuned_summary"] = pegasus_finetuned_summarize(text)
        except Exception as exc:
            result["pegasus_finetuned_summary"] = ""
            result["pegasus_finetuned_error"] = str(exc)

    run_default_generation = method in {"pegasus", "generate", "generation", "both", "randeng", "public", "all"}
    if run_default_generation:
        try:
            summary = randeng_pegasus_summarize(text)
            result["randeng_pegasus_summary"] = summary
            if method in {"pegasus", "generate", "generation", "both"}:
                result["generate_summary"] = summary
        except Exception as exc:
            result["randeng_pegasus_summary"] = ""
            result["randeng_pegasus_error"] = str(exc)
            if method in {"pegasus", "generate", "generation", "both"}:
                result["generate_summary"] = ""
                result["pegasus_error"] = str(exc)

    if method not in {
        "textrank",
        "pegasus",
        "pegasus_base",
        "pegasus_finetuned",
        "generate",
        "generation",
        "both",
        "all",
        "base",
        "zero_shot",
        "public",
        "randeng",
        "finetuned",
        "local",
    }:
        raise ValueError("method 只能是 textrank / pegasus_base / pegasus_finetuned / randeng / both / all")

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="摘要统一接口")
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--method", type=str, default="both")
    parser.add_argument("--model_name", type=str, default="randeng", help="保留兼容参数；推荐直接用 --method 指定模型")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(summarize(args.text, method=args.method, model_name=args.model_name), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
