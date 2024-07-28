import re
from dataclasses import dataclass
from functools import lru_cache
from itertools import chain
from typing import Optional

from LanguageTools.tokenizers.primitives import Token, extract_spans


def get_byte_to_char_map2(unicode_string):
    char_index = []
    byte_count = 0
    char_count = 0
    for char_offset, character in enumerate(unicode_string):
        bytes_ = character.encode('utf-8')
        for _ in bytes_:
            char_index.append(char_offset)
            byte_count += 1
        char_count += 1
    char_index.append(char_count)
    assert byte_count == len(unicode_string.encode("utf-8"))
    return char_index


@lru_cache
def create_subword_tokenizer(lang, vocab_size):
    from pathlib import Path
    try:
        from bpemb.util import sentencepiece_load, http_get
    except ImportError:
        raise ImportError("Install bpemb: pip install bpemb")

    def _load_file(file, archive=False):
        cache_dir = Path.home() / Path(".cache/bpemb")
        archive_suffix = ".tar.gz"
        base_url = "https://nlp.h-its.org/bpemb/"
        cached_file = Path(cache_dir) / file
        if cached_file.exists():
            return cached_file
        suffix = archive_suffix if archive else ""
        file_url = base_url + file + suffix
        print("downloading", file_url)
        return http_get(file_url, cached_file, ignore_tardir=True)
    model_file = "{lang}/{lang}.wiki.bpe.vs{vs}.model".format(lang=lang, vs=vocab_size)
    model_file = _load_file(model_file)
    spm = sentencepiece_load(model_file)
    return spm
    # return lambda text: spm.EncodeAsPieces(re.sub(r"\d", "0", text.lower()))


@dataclass
class TokenizerSpec:
    tokenizer_type: str
    lsp: Optional[str]
    lsp_id: Optional[int]
    tsp: Optional[str]
    tsp_id: Optional[int]
    unk: Optional[str]
    unk_id: Optional[int]


class SubwordTokenizer:
    tokenizer_types = {
        "bpemb": TokenizerSpec(tokenizer_type="sentencepiece", lsp=None, lsp_id=None, tsp=None, tsp_id=None, unk=None,
                               unk_id=None),
        "bert-base-uncased": TokenizerSpec(tokenizer_type="tokenizer", lsp="[CLS]", lsp_id=101, tsp="[SEP]", tsp_id=102,
                                           unk="[UNK]", unk_id=100),
        "bert-base-cased": TokenizerSpec(tokenizer_type="tokenizer", lsp="[CLS]", lsp_id=101, tsp="[SEP]", tsp_id=102,
                                         unk="[UNK]", unk_id=100),
        "bert-base-multilingual-uncased": TokenizerSpec(tokenizer_type="tokenizer", lsp="[CLS]", lsp_id=101,
                                                        tsp="[SEP]", tsp_id=102, unk="[UNK]", unk_id=100),
        "bert-base-multilingual-cased": TokenizerSpec(tokenizer_type="tokenizer", lsp="[CLS]", lsp_id=101, tsp="[SEP]",
                                                      tsp_id=102, unk="[UNK]", unk_id=100),
        "gpt2": TokenizerSpec(tokenizer_type="bpe", lsp=None, lsp_id=None, tsp=None, tsp_id=None, unk=None,
                              unk_id=None),
    }

    def __init__(self, tokenizer_name, **tokenizer_kwargs):
        if tokenizer_name == "bpemb":
            self.tokenizer = create_subword_tokenizer(lang="multi", vocab_size=1000000)

            # def tokenize_fn(text):
            #     return self.tokenizer.EncodeAsPieces(re.sub(r"\d", "0", text))
            #
            # self.tokenize_fn = tokenize_fn
            self.bpe = False
        elif tokenizer_name in self.tokenizer_types:
            if self.tokenizer_types[tokenizer_name].tokenizer_type == "tokenizer":
                try:
                    import tokenizers
                except ImportError:
                    raise ImportError("Install tokenizers: pip install tokenizers")

                self.tokenizer = tokenizers.Tokenizer.from_pretrained(tokenizer_name)
                self.subtokenize = None
            else:
                try:
                    import transformers
                    tokenizer_class_name = tokenizer_kwargs.get("tokenizer_class")
                    if tokenizer_class_name is None:
                        tokenizer_class_name = "AutoTokenizer"
                    if not hasattr(transformers, tokenizer_class_name):
                        raise ValueError(f"No such tokenizer in `transformers`: {tokenizer_class_name}")
                    tokenizer_class = getattr(transformers, tokenizer_class_name)
                except ImportError:
                    raise ImportError("Install transformers: pip install transformers")

                self.tokenizer = tokenizer_class.from_pretrained(tokenizer_name)

                if tokenizer_name.startswith("bert-base-multilingual"):
                    self.subtokenize = self._subtokenize_bert_multilingual
                elif tokenizer_name.startswith("bert-base"):
                    self.subtokenize = self._subtokenize_bert
                else:
                    self.subtokenize = None
        else:
            raise ValueError(f"Tokenizer not recognized: {tokenizer_name}")

        tokenizer_type = self.tokenizer_types[tokenizer_name].tokenizer_type
        if tokenizer_type == "tokenizer":
            self.align = self._align_with_tokenizer
            self.convert_tokens_to_string = self._tokenizer_tokens_to_string
            self.get_vocab = self._get_vocab_with_tokenizer
            self.tokenize_fn = self._tokenize_with_tokenizer
            self.prepare_subtokens = self._prepare_subtokens_with_tokenizer
        elif tokenizer_type == "sentencepiece":
            self.align = self._align_surface
            self.convert_tokens_to_string = self._sentencepiece_tokens_to_string
            self.get_vocab = self._get_vocab_sentencepiece
            self.tokenize_fn = self._tokenize_sentencepiece
            self.prepare_subtokens = self._prepare_subtokens_sentencepiece
        elif tokenizer_type == "wordpiece":
            self.align = self._align_wordpiece
            self.convert_tokens_to_string = self._wordpiece_tokens_to_string
            self.get_vocab = self._get_vocab_wordpiece
            self.tokenize_fn = self._tokenize_wordpiece
            self.prepare_subtokens = self._prepare_subtokens_wordpiece

            from LanguageTools.tokenizers import SimpleTokenizer
            self._pre_tokenizer = SimpleTokenizer()
        else:
            self.align = self._align_bpe
            self.convert_tokens_to_string = self._bpe_tokens_to_string
            self.get_vocab = self._get_vocab_bpe
            self.tokenize_fn = self._tokenize_bpe
            self.prepare_subtokens = self._prepare_subtokens_bpe

        if tokenizer_name == "bpemb":
            def tokenize_fn(text):
                return self.tokenizer.EncodeAsPieces(re.sub(r"\d", "0", text))

            self.tokenize_fn = tokenize_fn

    def _subtokenize_bert(self, text):
        return self.tokenizer.wordpiece_tokenizer.tokenize(text)

    def _subtokenize_bert_multilingual(self, text):
        # noinspection PyProtectedMember
        return [tt.value for tt in self.tokenizer._tokenizer.model.tokenize(text)]

    def _tokenize_with_tokenizer(self, text):
        return self.tokenizer.encode(text)

    def _tokenize_wordpiece(self, text):
        return self.tokenizer.encode(text)

    def _tokenize_sentencepiece(self, text):
        return self.tokenizer.tokenize(text)

    def _tokenize_bpe(self, text):
        return self.tokenizer.tokenize(text)

    @staticmethod
    def _align_with_tokenizer(tokens, text, lowercase=False):
        return tokens.offsets

    @staticmethod
    def _align_surface(tokens, text, lowercase=False):
        tokens_pos = 0
        in_token_pos = 0
        text_pos = 0

        text_len = len(text)

        if lowercase is True:
            text = text.lower()

        def get_current_token():
            nonlocal tokens_pos
            return tokens[tokens_pos]

        current_token = get_current_token()
        token_len = len(current_token)
        num_tokens = len(tokens)
        start = -1

        while text_pos < text_len:
            in_token_char = current_token[in_token_pos]
            in_text_char = text[text_pos]

            if in_token_char in {"▁"}:
                in_token_pos += 1
            elif current_token in {"[UNK]", "<unk>"}:
                raise NotImplemented
            else:
                if in_text_char != in_token_char:
                    text_pos += 1
                else:
                    if start == -1:
                        start = text_pos

                    in_token_pos += 1
                    text_pos += 1

            if in_token_pos == token_len:
                if start == -1:
                    start = text_pos
                yield start, text_pos
                start = -1
                in_token_pos = 0
                tokens_pos += 1
                if tokens_pos == num_tokens:
                    break
                current_token = get_current_token()
                token_len = len(current_token)

    @staticmethod
    def _align_wordpiece(tokens, text, lowercase=False):
        tokens_pos = 0
        subtoken_pos = 0
        in_token_pos = 0
        text_pos = 0

        text_len = len(text)

        if lowercase is True:
            text = text.lower()

        def get_current_token():
            nonlocal tokens_pos
            return tokens[tokens_pos][0]

        def get_current_subtokens():
            nonlocal tokens_pos
            return tokens[tokens_pos][1]

        def get_current_subtoken():
            nonlocal tokens_pos, subtoken_pos
            subtoken = tokens[tokens_pos][1][subtoken_pos]
            if subtoken.startswith("##"):
                return subtoken[2:]
            else:
                return subtoken

        current_token = get_current_token()
        current_subtokens = get_current_subtokens()
        current_subtoken = get_current_subtoken()
        subtokens_len = len(current_subtokens)
        subtoken_len = len(current_subtoken)

        num_tokens = len(tokens)
        start = -1

        while text_pos < text_len:
            in_token_char = current_subtoken[in_token_pos]
            in_text_char = text[text_pos]

            if current_subtoken in {"[UNK]"}:
                current_subtoken = current_token  # [UNK] is always the only subword for BERT
                subtoken_len = len(current_token)
                continue
            else:
                if in_text_char != in_token_char:
                    text_pos += 1
                else:
                    if start == -1:
                        start = text_pos

                    in_token_pos += 1
                    text_pos += 1

            if in_token_pos == subtoken_len:
                if start == -1:
                    start = text_pos
                yield start, text_pos

                start = -1
                in_token_pos = 0
                subtoken_pos += 1
                if subtoken_pos == subtokens_len:
                    subtoken_pos = 0
                    tokens_pos += 1
                if tokens_pos == num_tokens:
                    break

                current_token = get_current_token()
                current_subtokens = get_current_subtokens()
                current_subtoken = get_current_subtoken()
                subtokens_len = len(current_subtokens)
                subtoken_len = len(current_subtoken)

    @staticmethod
    def _align_bpe(tokens, text):
        _b2c = get_byte_to_char_map2(text)
        # _c2b = dict(zip(_b2c.values(), _b2c.keys()))

        cum_bytes = 0
        spans = []
        for token in tokens:
            token_len = len(token)
            token = token.lstrip("Ġ")
            token_len_no_space = len(token)

            spans.append((_b2c[cum_bytes + (token_len - token_len_no_space)], _b2c[cum_bytes + token_len]))
            cum_bytes += token_len

        return spans

    def _get_vocab_wordpiece(self):
        return self.tokenizer.vocab

    def _get_vocab_sentencepiece(self):
        raise NotImplemented

    def _get_vocab_bpe(self):
        raise NotImplemented

    def _get_vocab_with_tokenizer(self):
        return self.tokenizer.get_vocab()

    # noinspection PyMethodMayBeStatic
    def _prepare_subtokens_wordpiece(self, tokens):
        return list(chain(*(t[1] for t in tokens)))

    # noinspection PyMethodMayBeStatic
    def _prepare_subtokens_sentencepiece(self, tokens):
        return tokens

    # noinspection PyMethodMayBeStatic
    def _prepare_subtokens_bpe(self, tokens):
        return tokens

    def _prepare_subtokens_with_tokenizer(self, tokens):
        return tokens.tokens

    def token_gen(self, text):
        tokens = self.tokenize_fn(text)

        spans = self.align(tokens, text)
        subtokens = self.prepare_subtokens(tokens)

        last_token = None
        last_token_end = -1
        last_span = None
        generated_tokens = 0
        for ind, (token, span) in enumerate(zip(subtokens, spans)):
            if ind != 0:
                tailspace = span[0] - last_token_end  # text[last_token_end: span[0]]
                token_length = last_span[1] - last_span[0]
                yield Token(text=last_token, tailspace=tailspace, length=token_length, start=last_span[0])
                generated_tokens += 1

            last_token = token
            last_token_end = span[1]
            last_span = span

        yield Token(text=last_token, tailspace=0, length=last_span[1] - last_span[0], start=last_span[0])
        assert generated_tokens == len(subtokens) - 1

    def _tokenizer_tokens_to_string(self, tokens):
        return self.tokenizer.decode([self.tokenizer.token_to_id(token.text) for token in tokens])

    @staticmethod
    def _sentencepiece_tokens_to_string(tokens):
        # TODO
        # make this a separate function that does not require tokenizer instance
        return "".join(token.text + " " * token.tailspace for token in tokens).replace("▁", "").strip(" ")

    def _wordpiece_tokens_to_string(self, tokens):
        # TODO
        # make this a separate function that does not require tokenizer instance
        return self.tokenizer.convert_tokens_to_string([token.text for token in tokens])

    def _bpe_tokens_to_string(self, tokens):
        # TODO
        # make this a separate function that does not require tokenizer instance
        return self.tokenizer.convert_tokens_to_string([token.text for token in tokens]).strip(" ")
