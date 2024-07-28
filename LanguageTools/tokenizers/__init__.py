from LanguageTools.tokenizers.primitives import Doc, biluo_tags_from_offsets
from LanguageTools.tokenizers.utils import deaccent
from LanguageTools.tokenizers.subword import SubwordTokenizer
from LanguageTools.tokenizers.regexp import SimpleTokenizer


class Tokenizer:
    # noinspection PyShadowingNames
    def __init__(self, tokenizer_name=None, regexp=None, tokenizer_class=None):
        if tokenizer_name is None:
            tokenizer_name = "regexp"

        self._force_lower = "uncased" in tokenizer_name

        if tokenizer_name == "regexp":
            self._tokenizer = SimpleTokenizer(regexp=regexp)
        elif tokenizer_name in SubwordTokenizer.tokenizer_types:
            self._tokenizer = SubwordTokenizer(tokenizer_name, tokenizer_class=tokenizer_class)

    def __call__(self, text, lower=False, remove_accents=True):
        text = deaccent(text) if remove_accents else text
        text = text.lower() if lower or self._force_lower else text
        tokens = self._tokenizer.token_gen(text)
        return Doc(tokens)

    def get_vocab(self):
        if hasattr(self._tokenizer, "get_vocab"):
            return self._tokenizer.get_vocab()
        return None
