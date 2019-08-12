import re
from copy import copy
import sys

phonetic_map_path = sys.argv[1]#"ru_phonet_map_ext.txt"

def read_phonetic_map(path):
    ph_map = open(phonetic_map_path, "r")
    start = []
    middle = []
    end = []
    for line in ph_map.readlines():
        map_from, map_to = line.split()

        if map_to == "_":
            map_to = ""

        if map_from[-1] == "^":
            start.append((map_from[:-1], map_to))
        elif map_from[-1] == "$":
            end.append((map_from[:-1], map_to))
        else:
            middle.append((map_from, map_to))

    mapping = {"s":start, "e": end, "m": middle}

    return mapping


class PhoneticMapper:
    mapping = None

    def __init__(self):
        self.mapping = read_phonetic_map(phonetic_map_path)

    def __call__(self, text):
        text_mapped = ""
        for line in text.split("\n"):
            if line:
                for word in line.split():
                    mapped = self.map_word(word)
                    if mapped:
                        text_mapped += mapped + " "
                text_mapped += "\n"

        return text_mapped

    def map_word(self, word):
        word = word.upper()
        mapped = ""
        mapped_end = ""

        c_pos = 0
        word_end = len(word)

        for map_from, map_to in self.mapping["s"]:
            l = len(map_from)

            if word[c_pos: c_pos + l] == map_from:
                mapped += map_to
                c_pos += l
                break

        for map_from, map_to in self.mapping["e"]:
            l = len(map_from)

            if word[ - l:] == map_from:
                mapped_end += map_to
                word_end -= l
                break

        while c_pos < word_end:
            next_char = False
            for map_from, map_to in self.mapping["m"]:
                l = len(map_from)

                if word[c_pos: c_pos + l] == map_from:
                    mapped += map_to
                    c_pos += l
                    next_char = True
                    break

            if not next_char:
                mapped += word[c_pos]
                c_pos += 1

        mapped += mapped_end
        return mapped


text = """
Отпечатки падук и сама обувь являются объектом поклонения в индуизме, буддизме и джайнизме В современном обществе падуки значат намного больше нежели древнейшая обувь азиатского региона. Падуки стали культурным и религиозным символом Востока, участвующим в ритуалах и поклонении старейшинам, учителям, святым и божествам Символизм падук и их следов восходит в Индии к IV—V столетию Затем традиция поклонения им распространилась на Шри-Ланке, в Камбодже, Бирме и Таиланде Падука также участвует в особых торжественных церемониях, таких как свадьба Падука стала нарицательным именем и почётным титулом в некоторых государствах Юго-Восточной Азии
"""

text1 = """
Часов однообразный бой,
Томительная ночи повесть!
Язык для всех равно чужой
И внятный каждому, как совесть!

Кто без тоски внимал из нас,
Среди всемирного молчанья,
Глухие времени стенанья,
Пророчески-прощальный глас?
"""


if __name__ == "__main__":
    mapping = PhoneticMapper()

    # print(mapping(text1))

    import sys

    everything = []

    seen = set()
    
    for line in sys.stdin:

        word, normal, _ = line.strip().split()
        # count = int(count_str)

        ph_map = (normal, mapping.map_word(normal))

        seen.add(ph_map)

        # everything.append((word, mapping.map_word(word), count))
        # print(everything)
        # sys.exit()

        # print(line.split()[0], mapping(line), end="")

    everything = list(seen)

    everything.sort(key=lambda x: x[1])

    for e in everything:
        print("%s\t%s\t%s" % (e[0], e[1], phonetic_map_path.split(".")[0]))
