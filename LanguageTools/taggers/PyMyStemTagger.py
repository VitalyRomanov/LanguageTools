from pymystem3 import Mystem

class PyMyStemTagger:

    def __init__(self):
        self.tagger = Mystem()

    def parse(self, sentence):
        result = self.tagger.analyze(sentence)
        print(result)
        return [(t['text'].strip(), t['analysis'][0]['gr'] if 'analysis' in t and t['analysis'] else 'NONLEX') for t in result if t['text'].strip() not in {' ',''}]

    def tag_word(self, word):
        result = self.tagger.analyze(word)
        return [(t['text'].strip(), t['analysis'][0]['gr'] if 'analysis' in t and t['analysis'] else 'NONLEX') for t in result[:1] if t['text'].strip() not in {' ',''}]

if __name__=="__main__":
    tagger = PyMyStemTagger()
    print(tagger.parse("Кошек, в частности, слонов и носорогов В. Вавилов."))
    print(tagger.tag_word("кошек"))