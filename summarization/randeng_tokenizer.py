import collections
import os
import re
from typing import List, Optional, Tuple, Union

import jieba
from transformers import PreTrainedTokenizer
from transformers.models.bert.tokenization_bert import (
    BasicTokenizer,
    WordpieceTokenizer,
    _is_punctuation,
    load_vocab,
)


VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}


def _is_chinese_char(cp: int) -> bool:
    return (
        (0x4E00 <= cp <= 0x9FFF)
        or (0x3400 <= cp <= 0x4DBF)
        or (0x20000 <= cp <= 0x2A6DF)
        or (0x2A700 <= cp <= 0x2B73F)
        or (0x2B740 <= cp <= 0x2B81F)
        or (0x2B820 <= cp <= 0x2CEAF)
        or (0xF900 <= cp <= 0xFAFF)
        or (0x2F800 <= cp <= 0x2FA1F)
    )


class RandengPegasusTokenizer(PreTrainedTokenizer):
    """Tokenizer used by IDEA-CCNL Randeng Pegasus checkpoints.

    The upstream model repository ships a custom tokenizer file but does not
    expose it through AutoTokenizer metadata. This local version keeps the same
    vocabulary remapping and WordPiece behavior without depending on Fengshen.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str,
        do_lower_case: bool = True,
        do_basic_tokenize: bool = True,
        never_split: Optional[List[str]] = None,
        pad_token: str = "<pad>",
        eos_token: str = "</s>",
        unk_token: str = "<unk>",
        mask_token: str = "<mask_2>",
        mask_token_sent: str = "<mask_1>",
        additional_special_tokens: Optional[List[str]] = None,
        sep_token: str = "[SEP]",
        cls_token: str = "[CLS]",
        tokenize_chinese_chars: bool = True,
        strip_accents: Optional[bool] = None,
        offset: int = 100,
        **kwargs,
    ) -> None:
        if not os.path.isfile(vocab_file):
            raise ValueError(f"Can't find vocabulary file: {vocab_file}")

        self.offset = offset
        self.mask_token_sent = mask_token_sent
        self.vocab = load_vocab(vocab_file)
        self._rename_vocab_token("[unused1]", eos_token)
        self._rename_vocab_token("[PAD]", pad_token)
        self._rename_vocab_token("[UNK]", unk_token)
        if self.mask_token_sent is not None:
            self._rename_vocab_token("[unused3]", mask_token)
            self._rename_vocab_token("[unused2]", self.mask_token_sent)

        if additional_special_tokens is not None:
            if not isinstance(additional_special_tokens, list):
                raise TypeError("additional_special_tokens must be a list")
            additional_special_tokens = list(additional_special_tokens)
            if mask_token_sent and mask_token_sent not in additional_special_tokens:
                additional_special_tokens = [mask_token_sent] + additional_special_tokens
            additional_special_tokens += [
                f"<unk_{index}>" for index in range(len(additional_special_tokens), self.offset - 1)
            ]
        else:
            additional_special_tokens = [mask_token_sent] if mask_token_sent else []

        super().__init__(
            do_lower_case=do_lower_case,
            do_basic_tokenize=do_basic_tokenize,
            never_split=never_split,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            eos_token=eos_token,
            tokenize_chinese_chars=tokenize_chinese_chars,
            additional_special_tokens=additional_special_tokens,
            strip_accents=strip_accents,
            **kwargs,
        )

        self.pre_tokenizer = lambda text: jieba.cut(text, HMM=False)
        self.ids_to_tokens = collections.OrderedDict((ids, tok) for tok, ids in self.vocab.items())
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=self.unk_token)

    def _rename_vocab_token(self, old_token: str, new_token: str) -> None:
        if new_token in self.vocab:
            self.vocab.pop(old_token, None)
            return
        if old_token in self.vocab:
            self.vocab[new_token] = self.vocab.pop(old_token)
            return
        raise ValueError(f"Vocabulary is missing both {old_token!r} and {new_token!r}")

    @property
    def do_lower_case(self) -> bool:
        return self.basic_tokenizer.do_lower_case

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def get_vocab(self):
        return dict(self.vocab, **getattr(self, "added_tokens_encoder", {}))

    def _tokenize(self, text: str) -> List[str]:
        split_tokens = []
        for piece in self.pre_tokenizer(text):
            if piece in self.vocab:
                split_tokens.append(piece)
                continue
            if self.do_basic_tokenize:
                for token in self.basic_tokenizer.tokenize(piece, never_split=self.all_special_tokens):
                    if token in self.basic_tokenizer.never_split:
                        split_tokens.append(token)
                    else:
                        split_tokens.extend(self.wordpiece_tokenizer.tokenize(token))
            else:
                split_tokens.extend(self.wordpiece_tokenizer.tokenize(piece))
        return split_tokens

    def _convert_token_to_id(self, token: str) -> int:
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index: int) -> str:
        return self.ids_to_tokens.get(index, self.unk_token)

    @staticmethod
    def _cjk_punctuation() -> str:
        return (
            "\uff02\uff03\uff04\uff05\uff06\uff07\uff08\uff09\uff0a\uff0b\uff0c\uff0d"
            "\uff0f\uff1a\uff1b\uff1c\uff1d\uff1e\uff20\uff3b\uff3c\uff3d\uff3e"
            "\uff3f\uff40\uff5b\uff5c\uff5d\uff5e\uff5f\uff60\uff62\uff63\uff64"
            "\u3000\u3001\u3003\u3008\u3009\u300a\u300b\u300c\u300d\u300e\u300f"
            "\u3010\u3011\u3014\u3015\u3016\u3017\u3018\u3019\u301a\u301b\u301c"
            "\u301d\u301e\u301f\u3030\u303e\u303f\u2013\u2014\u2018\u2019\u201b"
            "\u201c\u201d\u201e\u201f\u2026\u2027\ufe4f\ufe51\ufe54\u00b7"
            "\uff01\uff1f\uff61\u3002"
        )

    def convert_ids_to_tokens(
        self,
        ids: Union[int, List[int]],
        skip_special_tokens: bool = False,
    ) -> Union[str, List[str]]:
        if isinstance(ids, int):
            if ids in self.added_tokens_decoder:
                return self.added_tokens_decoder[ids]
            return self._convert_id_to_token(ids)

        tokens = []
        for index in ids:
            index = int(index)
            if skip_special_tokens and index in self.all_special_ids and index != 2:
                continue
            if index in self.added_tokens_decoder:
                tokens.append(self.added_tokens_decoder[index])
            else:
                tokens.append(self._convert_id_to_token(index))
        return tokens

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        text = ""
        for index, token in enumerate(tokens):
            if token.startswith("##"):
                text += token[2:]
            elif len(token) == 1 and _is_chinese_char(ord(token)):
                text += token
            elif len(token) == 1 and _is_punctuation(token):
                text += token + " "
            elif index > 0 and text and _is_chinese_char(ord(text[-1])):
                text += token
            elif token == "</s>":
                continue
            else:
                text += " " + token

        text = re.sub(" +", " ", text)
        punctuation = re.sub(" +", "", self._cjk_punctuation()).strip() + "+-/={(<["
        punctuation_regex = "(%s) " % "|".join(re.escape(item) for item in punctuation)
        text = re.sub(punctuation_regex, "\\1", text)
        text = re.sub(r"(\d\.) (\d)", "\\1\\2", text)
        return text.strip()

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
    ) -> List[int]:
        if token_ids_1 is None:
            return token_ids_0 + [self.eos_token_id]
        return token_ids_0 + token_ids_1 + [self.eos_token_id]

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        all_special_ids = set(self.all_special_ids)
        if already_has_special_tokens:
            return [1 if token in all_special_ids else 0 for token in token_ids_0]
        if token_ids_1 is None:
            return [1 if token in all_special_ids else 0 for token in token_ids_0] + [1]
        return [1 if token in all_special_ids else 0 for token in token_ids_0 + token_ids_1] + [1]

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        return 1

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if os.path.isdir(save_directory):
            vocab_file = os.path.join(
                save_directory,
                (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"],
            )
        else:
            vocab_file = (filename_prefix + "-" if filename_prefix else "") + save_directory

        index = 0
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda item: item[1]):
                if index != token_index:
                    index = token_index
                writer.write(token + "\n")
                index += 1
        return (vocab_file,)
