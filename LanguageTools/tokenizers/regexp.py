import re

from LanguageTools.tokenizers.primitives import Doc, Token, extract_spans


class SimpleTokenizer:
    regexp = "(?:(?:https?|ftp):\/\/|\b(?:[a-z\d]+\.))(?:(?:[^\s()<>]+|\((?:[^\s()<>]+|(?:\([^\s()<>]+\)))?\))+(?:\((?:[^\s()<>]+|(?:\(?:[^\s()<>]+\)))?\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))?|[\w][\w-]+[\w]|[\w]\.|[\w]+|[^\w\s]|[0-9]+"
    empty = ()

    def __init__(self, regexp=None):
        if regexp is not None:
            self.regexp = regexp
        self.pattern = re.compile(self.regexp)
        self.tailing_space = " "
        self.no_tailing_space = ""

    def __call__(self, text, lower=False, remove_accents=True):
        tokens = self.token_gen(text)
        return Doc(tokens)

    def token_gen(self, text):
        match = self.pattern.search(text)

        if match is None: return iter(self.empty)  # return empty iterator if no tokens

        last_token = None; ends_at = 0
        last_token_span = None

        while match is not None:
            starts_at = match.start()
            # has_tailspace = ends_at != starts_at  # leading space for the current word
            tailspace = starts_at - ends_at  # text[ends_at: starts_at]
            ends_at = match.end()
            if last_token is not None:
                yield Token(text=last_token, tailspace=tailspace, length=len(last_token), start=last_token_span[0])
            last_token = text[starts_at: ends_at]
            last_token_span = (starts_at, ends_at)
            match = self.pattern.search(text, ends_at)

        yield Token(text=last_token, tailspace=0, length=len(last_token), start=last_token_span[0])

    @staticmethod
    def convert_tokens_to_string(tokens):
        return "".join(token.text + " " * token.tailspace for token in tokens).strip(" ")


def test_offset_extraction():
    text = '\rThis is usually how I do it.\tНо лучше не повторять.\n   Ciao.'

    offsets = [[9, 25], [30, 32], [42, 51]]
    tokenizer = SimpleTokenizer()
    tokens = list(tokenizer.token_gen(text))
    token_spans = extract_spans(tokens, offsets)
    for token_span, char_span in zip(token_spans, offsets):
        span_tokens = tokens[token_span[0]: token_span[1]]
        observed_span_str = tokenizer.convert_tokens_to_string(span_tokens)
        expected_span_str = text[char_span[0]: char_span[1]]
        assert observed_span_str == expected_span_str, f"{observed_span_str} != {expected_span_str}"


def test_tokenizer():
    text = '\rThis is usually how I do it.\tНо лучше не повторять.\n   Ciao.'
    tokenizer = SimpleTokenizer()

    tokens = list(tokenizer.token_gen(text))


if __name__ == "__main__":
    test_offset_extraction()
    test_tokenizer()
