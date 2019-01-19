import sys
sys.path.insert(0, "/Volumes/External/dev/")
from LanguageTools.CRFChunkParser import CRFChunkParser
from LanguageTools.PyMyStemTagger import PyMyStemTagger
from nltk import pos_tag
from nltk.tag.mapping import map_tag
from nltk import RegexpParser

from collections import Counter
import pickle

from nltk.chunk import tree2conlltags, conlltags2tree

def pick_tag(tags, pos):
    # return Counter(tags)

    first_choice = ""
    for tag, count in Counter(tags).most_common(2):
        if count > 2:
            return tag
        elif count == 2 and first_choice=="":
            first_choice = tag
        else:
            # print(pos[:2], pos[:2]=='A,', pos[:2]=='A=', pos[:2]=='S,', {'S,', 'A,','А='}, pos[2:] in set(['S,', 'A,','А=']))
            is_np_type = pos[:2] in {'S,', 'A,','А='} # does not work
            is_np_type = pos[:2]=='A,' or pos[:2]=='A=' or pos[:2]=='S,'
            o_is_candidate = "O" in {first_choice, tag}
            if not is_np_type and o_is_candidate:
                return "O"
            elif is_np_type and o_is_candidate:
                return first_choice if first_choice != "O" else tag # return the one that is not O
            else:
                return 'B-NP*'

class EnsembleCRFChunker:
    def __init__(self):
        self.chunker_nltk = CRFChunkParser([], model_file="/Users/LTV/dev/LanguageTools/russian_chunker_nltk.crf")
        self.chunker_ud = CRFChunkParser([], model_file="/Users/LTV/dev/LanguageTools/russian_chunker_ud.crf")
        self.chunker_mystem = CRFChunkParser([], model_file="/Users/LTV/dev/LanguageTools/russian_chunker_mystem.crf")
        self.chunker_iis = pickle.load(open("/Users/LTV/dev/LanguageTools/russian_chunker.pickle", "rb"))

        grammar = r"""
        NP:
            {<S.*|A(,|=).*>*<S.*>}  # Nouns and Adjectives, terminated with Nouns
        """
        self.grammar_chunker = RegexpParser(grammar)

        self.mystem_tagger = PyMyStemTagger()

    def parse(self, orig_tokens):
        if orig_tokens and type(orig_tokens[0]) is tuple:
            tokens = [token for token, _ in orig_tokens]
        else:
            tokens = orig_tokens
        
        tokenized_ud = list(map(lambda x: (x[0], map_tag('ru-rnc','universal', x[1])), pos_tag(tokens, lang='rus')))
        tokenized_nltk = pos_tag(tokens, lang='rus')
        tokenized_mystem = [(token, self.mystem_tagger.tag_word(token)[0][1]) for token in tokens]

        # print(self.chunker_iis.parse(tokenized_ud))

        tags_nltk = self.chunker_nltk.parse(tokenized_nltk, return_tree=False)
        tags_ud = self.chunker_nltk.parse(tokenized_ud, return_tree=False)
        tags_mystem = self.chunker_nltk.parse(tokenized_mystem, return_tree=False)
        tags_iis = tree2conlltags(self.chunker_iis.parse(tokenized_ud))
        tags_grammar = tree2conlltags(self.grammar_chunker.parse(tokenized_mystem))

        result_tags = [tags_nltk, tags_ud, tags_mystem, tags_iis, tags_grammar]

        if tokens is orig_tokens:
            tag_source = tags_ud
        else:
            tag_source = orig_tokens

        tags = [(token, tag_source[ind][1], pick_tag([tags_sp[ind][2] for tags_sp in result_tags], tags_mystem[ind][1])) for ind, token in enumerate(tokens)]

        for ind, (token,pos,iob_tag) in enumerate(tags):
            if token in set(['таких', 'такие', 'такими', 'как', 'включая', 'и', 'или','другие', 'других', 'другими', 'особенно', 'в', 'частности', ',']):
                tags[ind] = (token, pos, 'O')

        for ind, (token,pos,iob_tag) in enumerate(tags):
            if ind == 0:
                continue
            if iob_tag == "B-NP*":
                if tags[ind-1][2] in {'B-NP', 'I-NP'}:
                    tags[ind] = (token, pos, 'I-NP')
                else:
                    tags[ind] = (token, pos, 'B-NP')
            if iob_tag == "I-NP" and tags[ind-1][2] not in {'B-NP', 'I-NP'}:
                tags[ind] = (token, pos, 'B-NP')

        return conlltags2tree(tags)
        # return tags


if __name__=="__main__":
    chunker = EnsembleCRFChunker()
    from nltk import word_tokenize
    from pprint import pprint

    print(chunker.parse(word_tokenize("В частности, данные марки шашлыка сделаны из мяса высокого качества, в них нет даже остаточных следов антибиотиков, а также консервантов – бензойной и сорбиновой кислот – и фиксаторов окраски, усилителей вкуса и аромата, а также загустителей, — сообщает Роскачество.", "russian", preserve_line=True)))
    print(chunker.parse(word_tokenize("А потом, после того как был собран большой объем информации, мы поняли, что как в спортивной среде, так и среди военнослужащих, особенно тех, которые работают в сфере специальных операций (где присутствует постоянный риск для жизни), становятся популярными среди определенной части спортсменов и военнослужащих языческие идеи, идеи, проистекающие из языческого отношения к человеку", "russian", preserve_line=True)))
    print(chunker.parse(word_tokenize("Среди добровольцев был проведен опрос для оценки уровня употребления аспирина и таких нестероидных противовоспалительных препаратов (НПВП), как ибупрофен и напроксен.", "russian", preserve_line=True)))
    print(chunker.parse(word_tokenize("Волонтерский штаб в Одинцово уже посетили и записали видео обращение к Андрею Воробьеву такие известные люди как Александр Легков, Ольга Кармух на, Марина Юденич, Алексей Огурцов, Гузель Камаева, Елена Серова, Оксана Пушкина, Егор Кончаловский, Дмитрий Маликов, Дмитрий Губерниев, Денис Майданов, Джефф Монсон, главы муниципальных образований МО и депутаты Мособлдумы.", "russian", preserve_line=True)))
    



