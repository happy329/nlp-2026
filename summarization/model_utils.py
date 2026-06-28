import os

from huggingface_hub import hf_hub_download
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from common.text_utils import project_path
from summarization.randeng_tokenizer import RandengPegasusTokenizer


def resolve_model_path(model_name_or_path: str) -> str:
    local_path = project_path(model_name_or_path)
    if os.path.exists(local_path):
        return local_path
    return model_name_or_path


def load_summary_tokenizer(model_name_or_path: str):
    model_path = resolve_model_path(model_name_or_path)
    local_vocab = os.path.join(model_path, "vocab.txt") if os.path.isdir(model_path) else ""
    if local_vocab and os.path.exists(local_vocab):
        return RandengPegasusTokenizer(local_vocab)

    if "Randeng-Pegasus" in model_name_or_path:
        vocab_file = hf_hub_download(model_name_or_path, "vocab.txt")
        return RandengPegasusTokenizer(vocab_file)

    return AutoTokenizer.from_pretrained(model_path, use_fast=False)


def load_summary_model(model_name_or_path: str):
    return AutoModelForSeq2SeqLM.from_pretrained(resolve_model_path(model_name_or_path))
