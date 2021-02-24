import re
from gensim.utils import deaccent
from types import GeneratorType
from LanguageTools.wrappers.nltk_wrapper import Sentencizer

class Token:
    def __init__(self, text, tailspace, id=None, **kwargs):

        self.id = id
        self.text = text
        self.tailspace = tailspace

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        return repr(self.text)


class Doc:
    def __init__(self, tokens):
        if isinstance(tokens, list):
            self.tokens = tokens
        if isinstance(tokens, GeneratorType):
            self.tokens = list(tokens)

    def __iter__(self):
        return iter(self.tokens)

    def __len__(self):
        return len(self.tokens)

    def __repr__(self):
        return str(self)

    def __getitem__(self, item):
        return self.tokens[item]

    def __str__(self):
        str_ = ""
        for t in self.tokens:
            str_ += t.text
            if t.tailspace:
               str_ += " "
        return str_


class Tokenizer:
    regexp = "(?:(?:https?|ftp):\/\/|\b(?:[a-z\d]+\.))(?:(?:[^\s()<>]+|\((?:[^\s()<>]+|(?:\([^\s()<>]+\)))?\))+(?:\((?:[^\s()<>]+|(?:\(?:[^\s()<>]+\)))?\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))?|[\w][\w-]+[\w]|[\w]\.|[\w]+|[^\w\s]|[0-9]+"

    def __init__(self):
        self.pattern = re.compile(self.regexp)
        self.empty = ()
        self.tailing_space = " "
        self.no_tailing_space = ""

    def __call__(self, lines_str, lower = False, remove_accents=True):

        tokens = self.token_gen(lines_str, lower=lower, remove_accents=remove_accents)
        return Doc(tokens)

    def token_gen(self, lines_str, lower = False, remove_accents=True):

        lines = deaccent(lines_str.strip()) if remove_accents else lines_str
        lines = lines.lower() if lower else lines

        match = self.pattern.search(lines)

        if match is None: return iter(self.empty) # return empty iterator if no tokens

        last_token = None; ends_at = 0

        while match is not None:
            starts_at = match.start()
            tailspace = ends_at != starts_at # leading space for the current word
            ends_at = match.end()
            if last_token:
                # yield last_token, self.tailing_space if leading_space else self.no_tailing_space
                # yield last_token, leading_space
                # yield (last_token, leading_space) if mark_tailing_spaces else last_token
                yield Token(text=last_token, tailspace=tailspace)
            last_token = lines[starts_at:ends_at]
            match = self.pattern.search(lines, ends_at)

        # yield last_token, self.no_tailing_space
        # yield last_token, False
        # yield (last_token, False) if mark_tailing_spaces else last_token
        yield Token(text=last_token, tailspace=False)


if __name__ == "__main__":
    text_en = """<doc id="1300" url="https://en.wikipedia.org/wiki?curid=1300" title="Abalone">
Abalone

Abalone ( or ; via Spanish "", from the Rumsen language "aulón") is a common name for any of a group of small to very large sea snails, marine gastropod molluscs in the family Haliotidae.
Other common names are ear shells, sea ears, and muttonfish or muttonshells in Australia, ormer in Great Britain, perlemoen in South Africa, and in New Zealand.
S.W.A.T. M. D.
"""

    text_ru = """Агноёстици́зм (от др.-греч. ἄγνωστος — непознанный) — философская концепция, согласно которой мир непознаваем и люди не могут знать ничего достоверного о действительной сущности вещей; позиция религиозного агностицизма заключается в том, что люди не могут знать ничего достоверного о Боге (или богах)[1][2][3][4][5]. В общеупотребительном смысле (особенно в англоязычной литературе) агностицизм нередко смешивают с атеизмом, со скептицизмом в отношении религии вообще[1][2][4]. """

    from LanguageTools.wrappers.nltk_wrapper import Sentencizer
    sentencizer_en = Sentencizer("en")
    sentencizer_ru = Sentencizer("ru")
    tokenizer = Tokenizer()

    def test_tokenizer(sentencizer, tokenizer, text):
        for sent in sentencizer(text):
            tokenized = tokenizer(sent)
            print("s:", tokenized)
            print("s:", tokenized.tokens)
            print()

    for _ in range(1):
        test_tokenizer(sentencizer_en, tokenizer, text_en)
        test_tokenizer(sentencizer_ru, tokenizer, text_ru)