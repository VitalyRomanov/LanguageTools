from LanguageTools.tokenizers.primitives import Doc, biluo_tags_from_offsets
from LanguageTools.tokenizers.utils import deaccent
from LanguageTools.tokenizers.subword import SubwordTokenizer
from LanguageTools.tokenizers.regexp import SimpleTokenizer


class Tokenizer:
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


def test_tokenizer():
    text = '\rThis is usually how I do it.\tНо лучше не повторять.\n   Ciao.'
    nlp = Tokenizer("regexp")
    doc = nlp(text)
    assert (
        [tok.text for tok in doc.tokens] ==
        ['This', 'is', 'usually', 'how', 'I', 'do', 'it', '.', 'Но', 'лучше', 'не', 'повторять', '.', 'Ciao', '.']
    )

    text = '\rThis is usually how I do it.\tНо лучше не повторять.\n   Ciao.'
    nlp = Tokenizer("bert-base-uncased")
    doc = nlp(text)
    assert (
            [tok.text for tok in doc.tokens] ==
            ['this', 'is', 'usually', 'how', 'i', 'do', 'it', '.', 'н', '##о', 'л', '##у', '##ч', '##ш', '##е', 'н', '##е', 'п', '##ов', '##т', '##о', '##р', '##я', '##т', '##ь', '.', 'cia', '##o', '.']
    )

    text = '\rThis is usually how I do it.\tНо лучше не повторять.\n   Ciao.'
    nlp = Tokenizer("gpt2")
    doc = nlp(text)
    assert (
            [tok.text for tok in doc.tokens] ==
            ['č', 'This', 'Ġis', 'Ġusually', 'Ġhow', 'ĠI', 'Ġdo', 'Ġit', '.', 'ĉ', 'Ð', 'Ŀ', 'Ð¾', 'ĠÐ', '»', 'Ñĥ', 'Ñ', 'ĩ', 'Ñ', 'Ī', 'Ðµ', 'ĠÐ', '½', 'Ðµ', 'ĠÐ', '¿', 'Ð¾Ð', '²', 'ÑĤ', 'Ð¾', 'ÑĢ', 'Ñı', 'ÑĤ', 'ÑĮ', '.', 'Ċ', 'Ġ', 'Ġ', 'ĠC', 'iao', '.']
    )


def test_biluo():
    text = '\rThis is usually how I do it.\tНо лучше не повторять.\n   Ciao.'

    offsets = [[9, 25, "ent1"], [30, 32, "ent2"], [42, 51, "ent3"]]
    nlp = Tokenizer()
    doc = nlp(text)
    biluo = biluo_tags_from_offsets(doc, offsets)
    assert (
        biluo ==
        ['O', 'O', 'B-ent1', 'I-ent1', 'I-ent1', 'L-ent1', 'O', 'O', 'U-ent2', 'O', 'O', 'U-ent3', 'O', 'O', 'O']
    )


if __name__ == "__main__":
    test_biluo()
    test_tokenizer()
