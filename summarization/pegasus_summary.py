import argparse
import os
import sys
from functools import lru_cache
from typing import Iterable, List, Optional


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import config as cfg
from common.text_utils import clean_text, project_path
from summarization.model_utils import load_summary_model, load_summary_tokenizer


MODEL_ALIASES = {
    "base": cfg.pegasus_base_model,
    "zero_shot": cfg.pegasus_base_model,
    "public": cfg.public_summary_model,
    "randeng": cfg.public_summary_model,
    "finetuned": cfg.summary_model_dir,
    "local": cfg.summary_model_dir,
}


def require_transformers():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError("缺少 Pegasus 推理依赖，请安装: pip install transformers torch sentencepiece") from exc
    return torch


def resolve_model_name(model_name: str) -> str:
    model_name = MODEL_ALIASES.get(model_name, model_name)
    local_path = project_path(model_name)
    if os.path.exists(local_path):
        return local_path
    return model_name


class PegasusSummarizer:
    def __init__(
        self,
        model_name: str = "public",
        device: Optional[str] = None,
        max_source_len: int = cfg.summary_max_source_len,
        max_target_len: int = cfg.summary_max_target_len,
        min_target_len: int = cfg.summary_min_target_len,
        num_beams: int = cfg.summary_num_beams,
    ) -> None:
        self.model_name = resolve_model_name(model_name)
        self.device_name = device
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.min_target_len = min_target_len
        self.num_beams = num_beams
        self._torch = None
        self._tokenizer = None
        self._model = None
        self._device = None

    def load(self) -> None:
        if self._model is not None:
            return

        torch = require_transformers()
        self._torch = torch
        self._device = torch.device(self.device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._tokenizer = load_summary_tokenizer(self.model_name)
        self._model = load_summary_model(self.model_name)
        self._model.to(self._device)
        self._model.eval()

    def summarize(self, text: str) -> str:
        return self.summarize_batch([text])[0]

    def summarize_batch(self, texts: Iterable[str], batch_size: int = 4) -> List[str]:
        self.load()
        clean_texts = [clean_text(text) for text in texts]
        outputs = []

        for start in range(0, len(clean_texts), batch_size):
            batch = clean_texts[start : start + batch_size]
            inputs = self._tokenizer(
                batch,
                max_length=self.max_source_len,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(self._device) for key, value in inputs.items()}

            with self._torch.no_grad():
                generated = self._model.generate(
                    **inputs,
                    max_length=self.max_target_len,
                    min_length=self.min_target_len,
                    num_beams=self.num_beams,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )

            outputs.extend(
                self._tokenizer.batch_decode(
                    generated,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True,
                )
            )

        return [clean_text(item) for item in outputs]


@lru_cache(maxsize=4)
def get_summarizer(model_name: str = "public") -> PegasusSummarizer:
    return PegasusSummarizer(model_name=model_name)


def generate_summary(text: str, model_name: str = "public") -> str:
    return get_summarizer(model_name).summarize(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pegasus 生成式中文摘要")
    parser.add_argument("--text", type=str, required=True, help="待摘要文本")
    parser.add_argument(
        "--model_name",
        type=str,
        default="public",
        help="base/zero_shot/public/randeng/finetuned 或 HuggingFace/本地模型路径",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(generate_summary(args.text, model_name=args.model_name))


if __name__ == "__main__":
    main()
