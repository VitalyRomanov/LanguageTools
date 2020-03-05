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


np_tags = set(['B-NP', 'I-NP'])
def select_new_tag(p_tag, tag):
    new_tag = ""
    if p_tag not in np_tags:
            new_tag = 'O'
    elif p_tag == 'B-NP' and tag != 'I-NP':
        new_tag = 'U-NP'
    elif p_tag == 'B-NP' and tag == 'I-NP':
        new_tag = 'B-NP'
    elif p_tag == 'I-NP' and tag == 'I-NP':
        new_tag = 'I-NP'
    elif p_tag == 'I-NP' and tag != 'I-NP':
        new_tag = 'L-NP'
    return new_tag


def verify_tag(tag):
    return "O" if tag in ["PRT","LATN",""] else tag


def convert_scheme(sentence):
    previous = ()

    new_scheme = []
    for ind, (token, pos, tag) in enumerate(sentence):
        tag = verify_tag(tag)

        if ind == 0: 
            previous = (token, pos, tag)
            continue

        p_token, p_pos, p_tag = previous

        new_scheme += [(p_token, p_pos, select_new_tag(p_tag, tag))]

        previous = (token, pos, tag)

    p_token, p_pos, p_tag = previous
    new_scheme += [(token, pos, select_new_tag(p_tag, 'O'))]
    return new_scheme


def revert_scheme(sentence):
    new_scheme = []

    for token, pos, tag in sentence:
        if tag not in {'U-NP', 'L-NP'}:
            new_scheme += [(token, pos, tag)]
        elif tag == 'U-NP':
            new_scheme += [(token, pos, 'B-NP')]
        elif tag == 'L-NP':
            new_scheme += [(token, pos, 'I-NP')]

    return new_scheme


from nltk import CRFTagger

class CRFChunkParser2(ChunkParserI):
    def __init__(self, chunked_sents=[], feature_func=None, model_file=None, training_opt={}):
 
        # Transform the trees in IOB annotated sentences [(word, pos, chunk), ...]
        # chunked_sents = [tree2conlltags(sent) for sent in chunked_sents]
 
        # Transform the triplets in pairs, make it compatible with the tagger interface [((word, pos), chunk), ...]
        def triplets2tagged_pairs(iob_sent):
            return [((word, pos), chunk) for word, pos, chunk in iob_sent]
        
        chunked_sents = [convert_scheme(sent) for sent in chunked_sents]
        chunked_sents = [triplets2tagged_pairs(sent) for sent in chunked_sents]
 
        if feature_func is not None:
            feat_func = feature_func
        else:
            feat_func = self._feature_detector
        training_opt = {'feature.minfreq': 20, 'c2': 4.}
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
        iob_triplets = revert_scheme([(w, t, c) for ((w, t), c) in chunks])
 
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
            prevshape = ""
        elif index == 1:
            prevword = tokens[index - 1][0].lower()
            prevprevword = ""
            prevpos = simplify_pos(tokens[index - 1][1])
            prevprevpos = ""
            prevshape = ""
        else:
            prevword = tokens[index - 1][0].lower()
            prevprevword = tokens[index - 2][0].lower()
            prevpos = simplify_pos(tokens[index - 1][1])
            prevprevpos = simplify_pos(tokens[index - 2][1])
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

        def get_suffix_prefix(wordm, length):
            if len(word)>length:
                pref3 = word[:length].lower()
                suf3 = word[-length:].lower()
            else:
                pref3 = ""
                suf3 = ""
            return pref3, suf3

        suf_pref_lengths = [2,3]
        words = {
            'word': {'w': word, 'pos': pos, 'shape': shape(word)},
            'nword': {'w': nextword, 'pos': nextpos, 'shape': shape(nextword)},
            'nnword': {'w': nextnextword, 'pos': nextnextpos, 'shape': shape(nextnextword)},
            'pword': {'w': prevword, 'pos': prevpos, 'shape': shape(prevprevword)},
            'ppword': {'w': prevprevword, 'pos': prevprevpos, 'shape': shape(prevprevword)}
        }

        base_features = {}
        for word_position in words:
            for item in words[word_position]:
                base_features[word_position+"."+item] = words[word_position][item]

        prefix_suffix_features = {}
        for word_position in words:
            for l in suf_pref_lengths:
                feature_name_base = word_position+"."+repr(l)+"."
                pref, suf = get_suffix_prefix(words[word_position]['w'], l)
                prefix_suffix_features[feature_name_base+'pref'] = pref
                prefix_suffix_features[feature_name_base+'suf'] = suf
                prefix_suffix_features[feature_name_base+'pref.suf'] = '{}+{}'.format(pref, suf)
                prefix_suffix_features[feature_name_base+'posfix'] = '{}+{}+{}'.format(pref, words[word_position]['pos'], suf)
                prefix_suffix_features[feature_name_base+'shapefix'] = '{}+{}+{}'.format(pref, words[word_position]['shape'], suf)

        # pref3, suf3 = get_suffix_prefix(word)
        # prevpref3, prevsuf3 = get_suffix_prefix(prevword)
        # prevprevpref3, prevprevsuf3 = get_suffix_prefix(prevprevword)
        # nextpref3, nextsuf3 = get_suffix_prefix(nextword)
        # nextnextpref3, nextnextsuf3 = get_suffix_prefix(nextnextword)

        # postfix = '{}+{}+{}'.format(pref3, pos, suf3)

        # 89.6
        features = {
            # 'shape': shape(word),
            # 'wordlen': len(word),
            # 'prefix3': pref3,
            # 'suffix3': suf3,

            'pos': pos,
            'prevpos': prevpos,
            'nextpos': nextpos,
            'prevprevpos': prevprevpos,
            'nextnextpos': nextnextpos,

            # 'posfix': '{}+{}+{}'.format(pref3, pos, suf3),
            # 'prevposfix': '{}+{}+{}'.format(prevpref3, prevpos, prevsuf3),
            # 'prevprevposfix': '{}+{}+{}'.format(prevprevpref3, prevprevpos, prevprevsuf3),
            # 'nextposfix': '{}+{}+{}'.format(nextpref3, nextpos, nextsuf3),
            # 'nextnextposfix': '{}+{}+{}'.format(nextpref3, nextpos, nextsuf3),

            # 'word': word,
            # 'prevword': '{}'.format(prevword),
            # 'nextword': '{}'.format(nextword),
            # 'prevprevword': '{}'.format(prevprevword),
            # 'nextnextword': '{}'.format(nextnextword),

            # 'word+nextpos': '{0}+{1}'.format(postfix, nextpos),
            # 'word+nextnextpos': '{0}+{1}'.format(postfix, nextnextpos),
            # 'word+prevpos': '{0}+{1}'.format(postfix, prevpos),
            # 'word+prevprevpos': '{0}+{1}'.format(postfix, prevprevpos),
            
            'pos+nextpos': '{0}+{1}'.format(pos, nextpos),
            'pos+nextnextpos': '{0}+{1}'.format(pos, nextnextpos),
            'pos+prevpos': '{0}+{1}'.format(pos, prevpos),
            'pos+prevprevpos': '{0}+{1}'.format(pos, prevprevpos),
        }

        features.update(base_features)
        features.update(prefix_suffix_features)

        # return list(features.values())
        return features
 
 
if __name__ == "__main__":
    file_path = sys.argv[1]

    chunked_sents = [tree2conlltags(chunk.conllstr2tree(s)) for s in open(file_path).read().strip().split("\n\n")]

    random.shuffle(chunked_sents)

    train_sents = []#chunked_sents[:int(len(chunked_sents) * 0.7)]
    test_sents = chunked_sents[int(len(chunked_sents) * 0.7 + 1):]

    ### CRF Chunker
    chunker = CRFChunkParser2(chunked_sents=train_sents, model_file="russian_chunker.crf")
    print(chunker.evaluate([conlltags2tree(s) for s in test_sents]))

    ### Grammar chunker
    chunked_sents = [chunk.conllstr2tree(s, chunk_types=('NP',)) for s in open(file_path).read().strip().split("\n\n")]
    from nltk import RegexpParser
    grammar = r"""
        NP:
            {<S.*|A.*>*<S.*>}  # Nouns and Adjectives, terminated with Nouns
        """
    chunker = RegexpParser(grammar)
    print(chunker.evaluate(chunked_sents))
