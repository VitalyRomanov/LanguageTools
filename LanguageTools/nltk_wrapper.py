from nltk.tokenize import word_tokenize
from nltk.data import load
from nltk.tag import _get_tagger, _pos_tag

from nltk import RegexpParser
from nltk.chunk import tree2conlltags

from copy import copy

# from PatternDetector import en_keywords, ru_keywords

never_part_of_NP = set(["which", "thing", "many", "several", "few", "multiple", "all", "“", "”", "alike", "’", "–", "—", "overall", "this", "‘"])
key_tokens = set(["'s", "of", "in", "with", "for", "on", "over", "throughout"]) 
# key_tokens.update(en_keywords)
# key_tokens.update(ru_keywords)
key_tokens.update(never_part_of_NP)


en_grammar = r"""
        # NP:
        #     {<DET>?<NOUN|ADJ><NOUN|ADJ|'s|of|in|with|for|on|over|throughout>*<NOUN>}
        NP:
            {<DET>?<NOUN|ADJ|'s>*<NOUN>}
        NP_of:
            {<NP><of><NP>}
        NP_in:
            {<NP><in><NP>}
        NP_with:
            {<NP><with><NP>}
        NP_for:
            {<NP><for><NP>}
        NP_on:
        # maybe this rule should only work in subconcepts
            {<NP><on><NP>}
        NP_over:
            {<NP><over><NP>}
        NP_throughout:
            {<NP><throughout><NP>}
        """
        # TODO:
        # 1. add 's detection DONE
        # 2. handle a variant ’s DONE
        # 3. names do not seem to parse
        # 4. and NP does not process 100% of the time
        # 5. NP of NP
        # 6. Such thing, or no such thing are two antipatterns
        # 7. Incorporate numerals
        #       In addition to relatively young projects, a number of major exchanges have made their choice in favor of Malta, including Binance, OKEx, ZB.com, as well as such famous blockchain projects as TRON, Big One, Cubits, Bitpay and others.
        # 8. Antipatterns
        #       rid of NP


ru_grammar = r"""
        NP:
            {<NOUN|ADJ>*<NOUN.*>}
        """

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


class NltkWrapper:
    def __init__(self, language):
        self.lang_code = language
        self.lang_name = 'english' if language=='en' else 'russian'
        self.tagger_lang = 'eng' if language=='en' else 'rus'

        self.sent_tokenizer = load('tokenizers/punkt/{0}.pickle'.format(self.lang_name))
        self.tagger = _get_tagger(self.tagger_lang)

        self.grammar_parser = RegexpParser(en_grammar if language=="en" else ru_grammar)

    def sentencize(self, text):
        return self.sent_tokenizer.tokenize(text)

    def tokenize(self, sentence):
        tokens = word_tokenize(text=sentence, language=self.lang_name, preserve_line=True)

        tokens = process_apostrof_s(tokens)

        return tokens 

    def tag(self, tokens, tagset='universal', lang=None):
        return _pos_tag(tokens, tagset, self.tagger, self.tagger_lang)

    def __call__(self, text, tagger=True):
        sents = self.sentencize(text)
        t_sents = [self.tokenize(sent) for sent in sents]
        if tagger:
            tags = [self.tag(t_sent) for t_sent in t_sents]
        else:
            tags = t_sents
        # from nltk import pos_tag

        # tags = [pos_tag(sent, tagset='universal') for sent in sents]
        return tags

    def noun_chunks(self, tags):
        # t_sent = self.tokenize(sentence_text)
        # tags = self.tag(t_sent)
        for ind, tag in enumerate(tags):
            if tag[0].lower() in key_tokens:
                tags[ind] = (tags[ind][0], tags[ind][0].lower())
        # return [(token, pos, tag) for token, pos, tag in tree2conlltags(self.grammar_parser.parse(tags))]
        return self.grammar_parser.parse(tags)
        # return [[(token, tag) for token, pos, tag in tree2conlltags(self.grammar_parser.parse(tagged_tokens))] for tagged_tokens in tagged_sents]
        # return [tree2conlltags(self.grammar_parser.parse(tagged_tokens)) for tagged_tokens in tagged_sents]
        
    def chunks(self, tags):
        for ind, tag in enumerate(tags):
            if tag[0].lower() in key_tokens:
                tags[ind] = (tags[ind][0], tags[ind][0].lower())
        parsed = self.grammar_parser.parse(tags)
        return ["_".join([token for token, pos in s.leaves()]) for s in parsed.subtrees() if "NP" in s.label()]




if __name__=="__main__":
    from pprint import pprint
    nlp_en = NltkWrapper("en")
    text_en = "Alice`s Adventures in Wonderland (commonly shortened to Alice in Wonderland) is an 1865 novel written by English author Charles Lutwidge Dodgson under the pseudonym Lewis Carroll.[1] It tells of a young girl named Alice falling through a rabbit hole into a fantasy world populated by peculiar, anthropomorphic creatures."
    # text_en = "The kind of societal change Dr Ammous predicts in his book, and spoke about with their reporter, severely threatens a status quo, which such mainstream publications such as The Express is usually keen to uphold."
    # text_en = "This particular stream saw members of the community such as Bitcoin.com’s Roger Ver, Ethereum’s Vitalik Buterin who briefly visited, Andreas Brekken of Shitcoin.com, and many more special guests."
    # text_en = "With regard to Dai itself, stablecoin sceptics such as Preston Byrne often point out that the token is overcollateralized to ETH, so that creating $1 worth of Dai will take >$1 worth of ETH."
    text_en = "Its major dominance is in the Asian market especially South Korea, Singapore, and Japan."
    text_en = "The research will also include the Engineering, the Law School, School of Information, and other colleges or programs."
    # text_en = "Once launched, Huobi Chain will offer users a variety of benefits, including security, transparency, fast, scalability, and smart contract capability."
    tags_en = nlp_en(text_en)
    tags_en = nlp_en.noun_chunks(text_en)
    # tags_en.pprint()

    seen = []

    # print(tags_en.treepositions())

    from nltk import Tree
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
