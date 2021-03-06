#!/usr/bin/env python
# -*- coding: utf-8 -*-
from nltk import chunk
import sys
from nltk.chunk.named_entity import NEChunkParserTagger
from nltk.chunk import tree2conlltags, conlltags2tree
import random
import pickle
import re
from nltk import ChunkParserI, ClassifierBasedTagger


from nltk import CRFTagger

class CRFChunkParser(ChunkParserI):
    def __init__(self, chunked_sents=[], feature_func=None, model_file=None, training_opt={}):
 
        # Transform the trees in IOB annotated sentences [(word, pos, chunk), ...]
        # chunked_sents = [tree2conlltags(sent) for sent in chunked_sents]
 
        # Transform the triplets in pairs, make it compatible with the tagger interface [((word, pos), chunk), ...]
        def triplets2tagged_pairs(iob_sent):
            return [((word, pos), chunk) for word, pos, chunk in iob_sent]
        chunked_sents = [triplets2tagged_pairs(sent) for sent in chunked_sents]
 
        if feature_func is not None:
            feat_func = feature_func
        else:
            feat_func = self._feature_detector
        self.tagger = CRFTagger(feature_func=feat_func, training_opt=training_opt)
        if not model_file:
            raise Exception("Provide path to save model file")
        self.model_file = model_file
        if chunked_sents:
            self.train(chunked_sents)
        else:
            self.tagger.set_model_file(self.model_file)

    def train(self, chunked_sents):
        self.tagger.train(chunked_sents, self.model_file)
    
    def load(self, model_file):
        self.tagger.set_model_file(model_file)
 
    def parse(self, tagged_sent, return_tree = True):
        chunks = self.tagger.tag(tagged_sent)
 
        # Transform the result from [((w1, t1), iob1), ...] 
        # to the preferred list of triplets format [(w1, t1, iob1), ...]
        iob_triplets = [(w, t, c) for ((w, t), c) in chunks]
 
        # Transform the list of triplets to nltk.Tree format
        return conlltags2tree(iob_triplets) if return_tree else iob_triplets


    def _feature_detector(self, tokens, index):
        def shape(word):
            if re.match('[0-9]+(\.[0-9]*)?|[0-9]*\.[0-9]+$', word, re.UNICODE):
                return 'number'
            elif re.match('\W+$', word, re.UNICODE):
                return 'punct'
            elif re.match('\w+$', word, re.UNICODE):
                if word.istitle():
                    return 'upcase'
                elif word.islower():
                    return 'downcase'
                else:
                    return 'mixedcase'
            else:
                return 'other'


        def simplify_pos(s):
            if s.startswith('V'):
                return "V"
            else:
                return s.split('-')[0]

        word = tokens[index][0]
        pos = simplify_pos(tokens[index][1])
        if index == 0:
            prevword = prevprevword = ""
            prevpos = prevprevpos = ""
            prevshape = prevtag = prevprevtag = ""
        elif index == 1:
            prevword = tokens[index - 1][0].lower()
            prevprevword = ""
            prevpos = simplify_pos(tokens[index - 1][1])
            prevprevpos = ""
            prevtag = "" #history[index - 1][0]
            prevshape = prevprevtag = ""
        else:
            prevword = tokens[index - 1][0].lower()
            prevprevword = tokens[index - 2][0].lower()
            prevpos = simplify_pos(tokens[index - 1][1])
            prevprevpos = simplify_pos(tokens[index - 2][1])
            prevtag = "" #history[index - 1]
            prevprevtag = "" #history[index - 2]
            prevshape = shape(prevword)
        if index == len(tokens) - 1:
            nextword = nextnextword = ""
            nextpos = nextnextpos = ""
        elif index == len(tokens) - 2:
            nextword = tokens[index + 1][0].lower()
            nextpos = tokens[index + 1][1].lower()
            nextnextword = ""
            nextnextpos = ""
        else:
            nextword = tokens[index + 1][0].lower()
            nextpos = tokens[index + 1][1].lower()
            nextnextword = tokens[index + 2][0].lower()
            nextnextpos = tokens[index + 2][1].lower()

        # 89.6
        features = {
            'shape': '{}'.format(shape(word)),
            'wordlen': '{}'.format(len(word)),
            'prefix3': word[:3].lower(),
            'suffix3': word[-3:].lower(),
            'pos': pos,
            'word': word,
            # 'prevtag': '{}'.format(prevtag),
            'prevpos': '{}'.format(prevpos),
            'nextpos': '{}'.format(nextpos),
            'prevword': '{}'.format(prevword),
            'nextword': '{}'.format(nextword),
            'prevprevword': '{}'.format(prevprevword),
            'nextnextword': '{}'.format(nextnextword),
            'word+nextpos': '{0}+{1}'.format(word.lower(), nextpos),
            'word+nextnextpos': '{0}+{1}'.format(word.lower(), nextnextpos),
            'word+prevpos': '{0}+{1}'.format(word.lower(), prevpos),
            'word+prevprevpos': '{0}+{1}'.format(word.lower(), prevprevpos),
            'pos+nextpos': '{0}+{1}'.format(pos, nextpos),
            'pos+nextnextpos': '{0}+{1}'.format(pos, nextnextpos),
            'pos+prevpos': '{0}+{1}'.format(pos, prevpos),
            'pos+prevprevpos': '{0}+{1}'.format(pos, prevprevpos),
            # 'pos+prevtag': '{0}+{1}'.format(pos, prevtag),
            # 'shape+prevtag': '{0}+{1}'.format(prevshape, prevtag),
        }

        return list(features.values())
 
 
if __name__ == "__main__":

    # transformed = [list(map(lambda x: ((x[0], x[1]), x[2]), s)) for s in chunked_sents]
    # random.shuffle(transformed)
    # train_sents = transformed[:int(len(transformed) * 0.9)]
    # test_sents = transformed[int(len(transformed) * 0.9 + 1):]

    # from nltk.stem.snowball import SnowballStemmer

    file_path = sys.argv[1]
    chunked_sents = [tree2conlltags(chunk.conllstr2tree(s)) for s in open(file_path).read().strip().split("\n\n")]
    random.shuffle(chunked_sents)
    train_sents = []#chunked_sents[:int(len(chunked_sents) * 0.7)]
    test_sents = chunked_sents[int(len(chunked_sents) * 0.7 + 1):]

    ### CRF Chunker

    chunker = CRFChunkParser(chunked_sents=train_sents, model_file="russian_chunker.crf")
    print(chunker.evaluate([conlltags2tree(s) for s in test_sents]))


    # from nltk.tag.crf import CRFTagger
    # chunker = CRFTagger(feature_func=feature_detector)

    # chunker.set_model_file("russian_chunker.crf")
    # chunker.train(train_sents, "russian_chunker.crf")

    # from nltk.metrics import accuracy, precision, recall, f_measure
    # from itertools import chain
    # from nltk.tag.util import untag

    # def evaluate(chunker, gold):
    #     """
    #     Score the accuracy of the tagger against the gold standard.
    #     Strip the tags from the gold standard text, retag it using
    #     the tagger, then compute the accuracy score.

    #     :type gold: list(list(tuple(str, str)))
    #     :param gold: The list of tagged sentences to score the tagger on.
    #     :rtype: float
    #     """

    #     tagged_sents = chunker.tag_sents(untag(sent) for sent in gold)
    #     gold_tokens = list(chain(*gold))
    #     test_tokens = list(chain(*tagged_sents))
    #     return accuracy(gold_tokens, test_tokens), precision(set(gold_tokens), set(test_tokens)), recall(set(gold_tokens), set(test_tokens)), f_measure(set(gold_tokens), set(test_tokens))

    # print(evaluate(chunker, test_sents))
    # returns (0.957655158584272, 0.9429912369949786, 0.9362312229137476, 0.9395990712580529)
    # best result in https://www.nltk.org/book/ch07.html#fig-chunk-treerep is
    # 96.0%, 88.6%, 91.0%, 89.8%

    # test_line = """
    # Украинский боксер Александр Усик рассказал, что рассматривает возможность проведения боя с британцем Тони Белью, и готов для этого перейти в другой вес. Об этом сообщает корреспондент «Ленты.ру». «Я слышал, что Тони Белью хотел сразиться с победителем этого боя в тяжелом весе. Он недавно очень красиво победил Дэвида Хэя. От этого, я думаю, можно начинать. Я готов. Если он не хочет опускаться в мой вес, то я поднимусь в его», — сказал он. При этом боксер отметил, что нужно все взвесить: «Конечно, мне хочется больше кушать риса и макарон, но, может, еще есть хорошие и денежные бои в текущем весе». Гассиев, в свою очередь, назвал свой проигрыш отличным опытом, позволившим увидеть свои слабые стороны. «Мы вернемся намного сильнее», — пообещал он. 22 июля Усик единогласным решением судей выиграл поединок из 12 раундов против россиянина  Мурата Гассиева. Украинец стал победителем Всемирной боксерской суперсерии, а также обладателем четырех чемпионских поясов по версиям Международной боксерской федерации (IBF) и Всемирной боксерской ассоциации (WBA), Всемирной боксерской организации (WBO) и Всемирного боксерского совета (WBC).
    # """

    # from nltk import pos_tag, word_tokenize, sent_tokenize
    # pos_tok = [pos_tag(word_tokenize(s, "russian"), lang='rus') for s in sent_tokenize(test_line, "russian")]
    # print(pos_tok)
    # import pprint
    # pprint.pprint(chunker.tag_sents(pos_tok))

    ### Grammar chunker
    chunked_sents = [chunk.conllstr2tree(s, chunk_types=('NP',)) for s in open(file_path).read().strip().split("\n\n")]
    from nltk import RegexpParser
    grammar = r"""
        NP:
            {<S.*|A.*>*<S.*>}  # Nouns and Adjectives, terminated with Nouns
        """
    chunker = RegexpParser(grammar)
    print(chunker.evaluate(chunked_sents))
