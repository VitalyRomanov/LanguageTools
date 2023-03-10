import re
from dataclasses import dataclass
from functools import lru_cache
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


class SubwordTokenizer:
    tokenizer_types = {
        "bpemb": TokenizerSpec(tokenizer_type="sentencepiece", lsp=None, lsp_id=None, tsp=None, tsp_id=None),
        "bert-base-uncased": TokenizerSpec(tokenizer_type="wordpiece", lsp="[CLS]", lsp_id=101, tsp="[SEP]", tsp_id=102),
        "bert-base-cased": TokenizerSpec(tokenizer_type="wordpiece", lsp="[CLS]", lsp_id=101, tsp="[SEP]", tsp_id=102),
        "gpt2": TokenizerSpec(tokenizer_type="bpe", lsp=None, lsp_id=None, tsp=None, tsp_id=None),
    }

    def __init__(self, tokenizer_name, **tokenizer_kwargs):
        if tokenizer_name == "bpemb":
            self.tokenizer = create_subword_tokenizer(lang="multi", vocab_size=1000000)

            def tokenize_fn(text):
                return self.tokenizer.EncodeAsPieces(re.sub(r"\d", "0", text))

            self.tokenize_fn = tokenize_fn
            self.bpe = False
        elif tokenizer_name in self.tokenizer_types:
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

            def tokenize_fn(text):
                return self.tokenizer.tokenize(text)

            self.tokenize_fn = tokenize_fn
        else:
            raise ValueError(f"Tokenizer not recognized: {tokenizer_name}")


        tokenizer_type = self.tokenizer_types[tokenizer_name].tokenizer_type
        if tokenizer_type == "sentencepiece":
            self.align = self._align_surface
            self.convert_tokens_to_string = self._sentencepiece_tokens_to_string
        elif tokenizer_type == "wordpiece":
            self.align = self._align_surface
            self.convert_tokens_to_string = self._wordpiece_tokens_to_string
        else:
            self.align = self._align_bpe
            self.convert_tokens_to_string = self._bpe_tokens_to_string


    @staticmethod
    def _align_surface(tokens, text, lowercase=False):
        tokens_pos = 0
        in_token_pos = 0
        text_pos = 0

        text_len = len(text)

        if lowercase is True:
            text = text.lower()

        current_token = tokens[tokens_pos]
        token_len = len(current_token)
        num_tokens = len(tokens)
        start = -1

        while text_pos < text_len:
            in_token_char = current_token[in_token_pos]
            in_text_char = text[text_pos]
            if in_token_char in {"#", "▁"}:
                in_token_pos += 1
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
                yield (start, text_pos)
                start = -1
                in_token_pos = 0
                tokens_pos += 1
                if tokens_pos == num_tokens:
                    break
                current_token = tokens[tokens_pos]
                token_len = len(current_token)

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

    def token_gen(self, text):
        tokens = self.tokenize_fn(text)
        # tokens = self.tokenize_fn(re.sub(r"\d", "0", text))

        spans = self.align(tokens, text)
        last_token = None
        last_token_end = -1
        last_span = None
        generated_tokens = 0
        for ind, (token, span) in enumerate(zip(tokens, spans)):
            if ind != 0:
                tailspace = span[0] - last_token_end  # text[last_token_end: span[0]]
                token_length = last_span[1] - last_span[0]
                yield Token(text=last_token, tailspace=tailspace, length=token_length, start=last_span[0])
                generated_tokens += 1

            last_token = token
            last_token_end = span[1]
            last_span = span

        yield Token(text=last_token, tailspace=0, length=last_span[1] - last_span[0], start=last_span[0])
        assert generated_tokens == len(tokens) - 1

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


def verify_tokens(text, tokens, spans, do_assert=True):
    last_token_end = None
    for ind, (t, sp) in enumerate(zip(tokens, spans)):
        expected_token = t.strip("▁").strip("#").replace("č", "\r").replace("ĉ", "\t").replace("Ċ", "\n")
        observed_token = text[sp[0]: sp[1]]
        if do_assert:
            assert expected_token == observed_token
        if ind > 0:
            observed_space = text[last_token_end: sp[0]]
        else:
            observed_space = ""
        last_token_end = sp[1]
        # print(repr(expected_token), repr(observed_token), repr(observed_space))
    # print()


def test_alignment():
    text = '\rThis is usually how I do it.\tНо лучше не повторять.\n   Ciao.'

    tokens = ['▁', '\rT', 'his', '▁is', '▁usually', '▁how', '▁', 'I', '▁do', '▁it', '.', '\tН', 'о', '▁лучше', '▁не', '▁повторя', 'ть', '.', '\n', '▁', 'C', 'iao', '.']
    spans = SubwordTokenizer._align_surface(
        tokens=tokens,
        text=text
    )
    verify_tokens(text, tokens, spans)

    tokens = ['This', 'is', 'usually', 'how', 'I', 'do', 'it', '.', 'Н', '##о', 'л', '##у', '##ч', '##ш', '##е', 'н', '##е', 'п', '##ов', '##т', '##о', '##р', '##я', '##т', '##ь', '.', 'C', '##ia', '##o', '.']
    spans = SubwordTokenizer._align_surface(
        tokens=tokens,
        text=text
    )
    verify_tokens(text, tokens, spans)

    tokens = ['č', 'This', 'Ġis', 'Ġusually', 'Ġhow', 'ĠI', 'Ġdo', 'Ġit', '.', 'ĉ', 'Ð', 'Ŀ', 'Ð¾', 'ĠÐ', '»', 'Ñĥ', 'Ñ', 'ĩ', 'Ñ', 'Ī', 'Ðµ', 'ĠÐ', '½', 'Ðµ', 'ĠÐ', '¿', 'Ð¾Ð', '²', 'ÑĤ', 'Ð¾', 'ÑĢ', 'Ñı', 'ÑĤ', 'ÑĮ', '.', 'Ċ', 'Ġ', 'Ġ', 'ĠC', 'iao', '.']
    spans = SubwordTokenizer._align_bpe(
        tokens=tokens,
        text=text
    )
    verify_tokens(text, tokens, spans, do_assert=False)


def test_offset_extraction():
    text = '\rThis is usually how I do it.\tНо лучше не повторять.\n   Ciao.'

    offsets = [[9, 25], [29, 32], [42, 51]]
    tokenizer = SubwordTokenizer("bpemb")
    tokens = list(tokenizer.token_gen(text))
    token_spans = extract_spans(tokens, offsets)
    for token_span, char_span in zip(token_spans, offsets):
        span_tokens = tokens[token_span[0]: token_span[1]]
        observed_span_str = tokenizer.convert_tokens_to_string(span_tokens)
        expected_span_str = text[char_span[0]: char_span[1]]
        assert observed_span_str == expected_span_str, f"{observed_span_str} != {expected_span_str}"

    offsets = [[9, 25], [30, 32], [42, 51]]
    tokenizer = SubwordTokenizer("bert-base-cased", tokenizer_class="BertTokenizer")
    tokens = list(tokenizer.token_gen(text))
    token_spans = extract_spans(tokens, offsets)
    for token_span, char_span in zip(token_spans, offsets):
        span_tokens = tokens[token_span[0]: token_span[1]]
        observed_span_str = tokenizer.convert_tokens_to_string(span_tokens)
        expected_span_str = text[char_span[0]: char_span[1]]
        assert observed_span_str == expected_span_str

    offsets = [[9, 25], [30, 32], [42, 51]]
    tokenizer = SubwordTokenizer("gpt2")
    tokens = list(tokenizer.token_gen(text))
    token_spans = extract_spans(tokens, offsets)
    for token_span, char_span in zip(token_spans, offsets):
        span_tokens = tokens[token_span[0]: token_span[1]]
        observed_span_str = tokenizer.convert_tokens_to_string(span_tokens)
        expected_span_str = text[char_span[0]: char_span[1]]
        assert observed_span_str == expected_span_str


def test_bpe_tokenizer():
    text = '\rThis is usually how I do it.\tНо лучше не повторять.\n   Ciao.'
    tokenizer = SubwordTokenizer("bpemb")
    tokens = list(tokenizer.token_gen(text))

    tokenizer = SubwordTokenizer("bert-base-cased", tokenizer_class="BertTokenizer")
    tokens = list(tokenizer.token_gen(text))

    tokenizer = SubwordTokenizer("gpt2")
    tokens = list(tokenizer.token_gen(text))


if __name__ == "__main__":
    test_alignment()
    test_offset_extraction()
    test_bpe_tokenizer()
