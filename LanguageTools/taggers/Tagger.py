import pymorphy2

class Tagger:

    def __init__(self):
        self.morph = pymorphy2.MorphAnalyzer()

    def __call__(self, text):
        text_mapped = ""
        for line in text.split("\n"):
            if line:
                for word in line.split():
                    mapped = self.tag_word(word)
                    if mapped:
                        text_mapped += mapped + " "
                text_mapped += "\n"

        return text_mapped

    def tag_word(self, word):
        parse = self.morph.parse(word)
        tag = parse[0].tag.POS
        if tag is None:
            return word
        else:
            return word + "_" + tag

    def get_lemma(self, word):
        parse = self.morph.parse(word)
        return parse[0].normal_form


# text1 = """
# Часов однообразный бой,
# Томительная ночи повесть!
# Язык для всех равно чужой
# И внятный каждому, как совесть!

# Кто без тоски внимал из нас,
# Среди всемирного молчанья,
# Глухие времени стенанья,
# Пророчески-прощальный глас?
# """


# if __name__ == "__main__":
#     mapping = PosTagger()

#     print(mapping(text1))
