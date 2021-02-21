from abc import ABC, abstractmethod
from typing import Tuple, List

from nltk import RegexpParser
from nltk import Tree
from nltk.data import load
from nltk.tag import _get_tagger, _pos_tag
from nltk.tokenize import word_tokenize

from LanguageTools.wrappers.grammars import EnNpGrammar, RuNpGrammar


class NpEnExceptions:
    help_avoiding_NP = {'Dr', 'dr', 'kind', 'of', 'etc'}
    exceptions_ = {"which", "thing", "many", "several", "few", "multiple", "all", "“", "”", "alike", "’", "–", "—",
                  "overall", "this", "‘"}
    keywords = help_avoiding_NP | exceptions_


class NpRuExceptions:
    help_avoiding_NP = {'на', 'качестве', 'том', 'числе', 'под', 'руководством', 'т.', 'д.', 'п.'}
    exceptions_ = {"различные", "различных", "различным", "различными"}
    keywords = help_avoiding_NP | exceptions_


class NpExceptions:
    key_tokens = NpEnExceptions.keywords | NpEnExceptions.keywords


def process_apostrof_s(tokens):
    # tokens = copy(tokens)
    locations = []
    for ind, token in enumerate(tokens):
        if ind == len(tokens) - 1:
            continue
        
        if (token == "`" or token == "’" or token == "‘") and tokens[ind+1] == "s":
            locations.append(ind)

    while locations:
        c_loc = locations.pop(-1)
        tokens.pop(c_loc)
        tokens.pop(c_loc)
        tokens.insert(c_loc, "'s")

    return tokens


class Sentencizer:
    def __init__(self, lang):
        lang = lang.lower()
        if lang in {'rus', "ru", "russian", ''}:
            language = 'russian'
        elif lang in {'eng', 'en', 'english'}:
            language = 'english'
        else:
            raise NotImplementedError(f"Language is not supported: {lang}")

        self._tokenizer = load("tokenizers/punkt/{0}.pickle".format(language))

    def __call__(self, text):
        return self._tokenizer.tokenize(text)


class PosTagger:
    def __init__(self, lang):
        lang = lang.lower()
        if lang in {'rus', "ru", "russian", ''}:
            language = 'rus'
        elif lang in {'eng', 'en', 'english'}:
            language = 'eng'
        else:
            raise NotImplementedError(f"Language is not supported: {lang}")
        self._tagger_lang = language
        self._tagger = _get_tagger(language)

    def __call__(self, tokens, tagset='universal'):
        return _pos_tag(tokens, tagset, self._tagger, self._tagger_lang)


class GrammarParser(ABC):
    def __init__(self, lang):
        lang = lang.lower()
        if lang in {'rus', "ru", "russian", ''}:
            language = 'ru'
        elif lang in {'eng', 'en', 'english'}:
            language = 'en'
        else:
            raise NotImplementedError(f"Language is not supported: {lang}")

        self._grammar_parser = RegexpParser(self._get_grammar(language))

    @abstractmethod
    def _get_grammar(self, lang):
        pass

    def _set_exception_set(self):
        pass

    def _recover_tags(self, tree, mapping):
        for pos in range(len(tree)):
            position = tree[pos]
            if isinstance(position, Tree):
                self._recover_tags(position, mapping)
            else:
                tree[pos] = (position[0], mapping.get(position[0], position[1]))

        return tree

    def __call__(self, tagged_tokens: List[Tuple[str, str]]):
        replacements = {}

        for ind, tag in enumerate(tagged_tokens):
            if tag[0].lower() in NpExceptions.key_tokens:
                replacements[tagged_tokens[ind][0]] = tagged_tokens[ind][1]
                tagged_tokens[ind] = (tagged_tokens[ind][0], tagged_tokens[ind][0].lower())
        return self._recover_tags(self._grammar_parser.parse(tagged_tokens), replacements)


class NpParser(GrammarParser):
    def __init__(self, lang):
        super(NpParser, self).__init__(lang)

    def _get_grammar(self, lang):
        if lang == "en":
            return EnNpGrammar.grammar
        elif lang == "ru":
            return RuNpGrammar.grammar
        else:
            raise NotImplementedError(f"Language is not supported: {lang}")

    def _set_exception_set(self):
        self.exceptions = NpExceptions.key_tokens


class NltkWrapper:
    def __init__(self, language):
        self.lang_code = language
        self.lang_name = 'english' if language=='en' else 'russian'
        self.tagger_lang = 'eng' if language=='en' else 'rus'

        self.sent_tokenizer = Sentencizer(self.lang_name)
        self.tagger = PosTagger(self.tagger_lang)

        self.grammar_parser = NpParser(self.lang_name)

    def sentencize(self, text):
        return self.sent_tokenizer(text)

    def preprocess(self, text):
        return text

    def tokenize(self, sentence):
        sentence = self.preprocess(sentence)
        tokens = word_tokenize(text=sentence, language=self.lang_name, preserve_line=True)

        tokens = process_apostrof_s(tokens)

        return tokens 

    def tag(self, tokens, tagset='universal'):
        return self.tagger(tokens, tagset=tagset)

    def __call__(self, text, tagger=True):
        sents = self.sentencize(text)
        t_sents = [self.tokenize(sent) for sent in sents]
        if tagger:
            tags = [self.tag(t_sent) for t_sent in t_sents]
        else:
            tags = t_sents
        return tags

    def parse_grammar(self, tokens):
        return self.grammar_parser(tokens)

    def chunks(self, tags):
        parsed = self.parse_grammar(tags)
        return ["_".join([token for token, pos in s.leaves()]) for s in parsed.subtrees() if "NP" in s.label()]


if __name__=="__main__":
    nlp_en = NltkWrapper("en")
    text_en = "Alice`s Adventures in Wonderland (commonly shortened to Alice in Wonderland) is an 1865 novel written by English author Charles Lutwidge Dodgson under the pseudonym Lewis Carroll.[1] It tells of a young girl named Alice falling through a rabbit hole into a fantasy world populated by peculiar, anthropomorphic creatures."
    # text_en = "The kind of societal change Dr Ammous predicts in his book, and spoke about with their reporter, severely threatens a status quo, which such mainstream publications such as The Express is usually keen to uphold."
    # text_en = "This particular stream saw members of the community such as Bitcoin.com’s Roger Ver, Ethereum’s Vitalik Buterin who briefly visited, Andreas Brekken of Shitcoin.com, and many more special guests."
    # text_en = "With regard to Dai itself, stablecoin sceptics such as Preston Byrne often point out that the token is overcollateralized to ETH, so that creating $1 worth of Dai will take >$1 worth of ETH."
    text_en = "Its major dominance is in the Asian market especially South Korea, Singapore, and Japan."
    text_en = "The research will also include the Engineering, the Law School, School of Information, and other colleges or programs."
    # text_en = "Once launched, Huobi Chain will offer users a variety of benefits, including security, transparency, fast, scalability, and smart contract capability."
    tags_en = nlp_en(text_en)
    for sent in tags_en:
        tags_en = nlp_en.chunks(sent)
        tags_en.pprint()

    seen = []

    # print(tags_en.treepositions())

    for child in tags_en:
        if isinstance(child, Tree):
            if len(child.label()) > 3 and child.label()[:3] == "NP_":
                for c in child:
                    print("\t", c)
        else:
            print(child)

        # print(child)

    # for tree in tags_en.subtrees():
    #     if tree.label == "S":
    #         pass
    #     else:
    #         if len(tree.label()) > 3 and tree.label()[:3] == "NP_":
    #             nested = []
    #             nested.append(" ".join([tok for tok, _ in tree.leaves()]))
    #             print(nested)
            
            # nested.append([t for t in tree.subtrees() if t.label() == "NP"])

    # all_trees = [tree for tree in tags_en.subtrees() if tree not in nested]

    # for tree in all_trees:

    #     # for t in tree:
    #     tree.pprint()
    #     # print(list(tree.subtrees()))

    #     print("\n\n\n")
    # pprint(tags_en)

    # nlp_ru = NltkWrapper("ru")
    # text_ru = "«Приключения Алисы в Стране чудес» (англ. Alice’s Adventures in Wonderland), часто используется сокращённый вариант «Алиса в Стране чудес» (англ. Alice in Wonderland) — сказка, написанная английским математиком, поэтом и прозаиком Чарльзом Лютвиджем Доджсоном под псевдонимом Льюис Кэрролл и изданная в 1865 году. В ней рассказывается о девочке по имени Алиса, которая попадает сквозь кроличью нору в воображаемый мир, населённый странными антропоморфными существами."
    # tags_ru = nlp_ru(text_ru)
    # tags_ru = nlp_en.noun_chunks(text_ru)
    # pprint(tags_ru)
