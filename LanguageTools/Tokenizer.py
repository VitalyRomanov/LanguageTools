import sys
import os
import pickle
import mmap
import re
from collections import Counter
from nltk import RegexpTokenizer
from gensim.utils import deaccent
# from collections import namedtuple
from types import GeneratorType

# Token = namedtuple("Token", "text tailspace", defaults=(None, None))

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


# nltk_regexp = "[A-Za-zА-Яа-яё]\.|[A-Za-zА-Яа-яё][A-Za-zА-Яа-яё-]*|[^\w\s]|[0-9]+"
# nltk_regexp = "(?:(?:https?|ftp):\/\/|\b(?:[a-z\d]+\.))(?:(?:[^\s()<>]+|\((?:[^\s()<>]+|(?:\([^\s()<>]+\)))?\))+(?:\((?:[^\s()<>]+|(?:\(?:[^\s()<>]+\)))?\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))?|[A-Za-zА-Яа-яё]\.|[A-Za-zА-Яа-яё][A-Za-zА-Яа-яё-]+[A-Za-zА-Яа-яё]|[A-Za-zА-Яа-яё]+|[\w]+|[^\w\s]|[0-9]+"
nltk_regexp = "(?:(?:https?|ftp):\/\/|\b(?:[a-z\d]+\.))(?:(?:[^\s()<>]+|\((?:[^\s()<>]+|(?:\([^\s()<>]+\)))?\))+(?:\((?:[^\s()<>]+|(?:\(?:[^\s()<>]+\)))?\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))?|[\w][\w-]+[\w]|[\w]\.|[\w]+|[^\w\s]|[0-9]+"
# "[A-Za-zА-Яа-яё][A-Za-zА-Яа-яё-]+|[^\w\s]|[0-9]+"

# punct_chars = "[-!\"#$%&'()*+,/:;<=>?@[\]^_`{|}~—»«“”„….]" # add point .

# punct_chars_point = "[-!\"#$%&'()*+,/:;<=>?@[\]^_`{|}~—»«“”„….]" # add point .

# punct = re.compile(punct_chars)

# def keep_hyphen(search_str, position):
#     if search_str[position] != "-":
#         return False
#     if search_str[position] == "-" and \
#             (position + 1 > len(search_str)):
#         return False
#     if search_str[position] == "-" and \
#             (position + 1 < len(search_str)) and \
#                 (search_str[position + 1] in punct_chars or search_str[position + 1] == " "):
#         return False
#     return True

# def expandall(sent_orig):
#     pos = 0

#     # replace illusive space
#     # sent = sent_orig.replace(" ", " ") # added to replace_accents_rus
#     sent = replace_accents_rus(sent)

#     new_str = ""
#     search_str = sent[0:]
#     res = re.search(punct, search_str)

#     while res is not None:
#         begin_at = res.span()[0]
#         end_at = res.span()[1]

#         new_str += search_str[:begin_at]

#         if len(new_str) > 0 and \
#             begin_at != 0 and \
#                 search_str[begin_at] != "-" and \
#                     new_str[-1] != " " and \
#                         not keep_hyphen(search_str, begin_at): # some problem here << didn't detect --.
#             new_str += " "
#         new_str += search_str[begin_at]

#         if len(search_str) > end_at and \
#                 search_str[begin_at] != "-" and \
#                     search_str[end_at] != " ":
#             new_str += " "

#         if len(search_str) > end_at:
#             search_str = search_str[end_at:]
#         else:
#             search_str = ""
#         res = re.search(punct, search_str)
#     new_str += search_str


#     return new_str

def replace_accents_rus(sent_orig):

    sent = sent_orig.replace("о́", "о")
    sent = sent.replace("а́", "а")
    sent = sent.replace("е́", "е")
    sent = sent.replace("у́", "у")
    sent = sent.replace("и́", "и")
    sent = sent.replace("ы́", "ы")
    sent = sent.replace("э́", "э")
    sent = sent.replace("ю́", "ю")
    sent = sent.replace("я́", "я")
    sent = sent.replace("о̀", "о")
    sent = sent.replace("а̀", "а")
    sent = sent.replace("ѐ", "е")
    sent = sent.replace("у̀", "у")
    sent = sent.replace("ѝ", "и")
    sent = sent.replace("ы̀", "ы")
    sent = sent.replace("э̀", "э")
    sent = sent.replace("ю̀", "ю")
    sent = sent.replace("я̀", "я")
    sent = sent.replace(b"\u0301".decode('utf8'), "")
    sent = sent.replace(b"\u00AD".decode('utf8'), "")
    sent = sent.replace(b"\u00A0".decode('utf8'), " ")
    sent = sent.replace(" ", " ")
    return sent


# class Tokenizer:
#     def __call__(self,lines, lower = False, split = True):
#         lines = lines.strip().split("\n")
#         tokenized = ""
#         for line in lines:
#             if lower:
#                 tokenized += expandall(line.lower())
#             else:
#                 tokenized += expandall(line)
#             # if len(lines) > 1:
#             #     tokenized += " N "
#         if split:
#             return tokenized.split()
#         else:
#             return tokenized


class Tokenizer:

    def __init__(self):
        self.tokenizer = RegexpTokenizer(nltk_regexp)
        self.pattern = re.compile(nltk_regexp)
        self.empty = ()
        self.tailing_space = " "
        self.no_tailing_space = ""

    # def __call__(self, lines_str, lower = False, split = True, remove_accents=True, mark_tailing_spaces=False):
    #     lines = deaccent(lines_str.strip()) if remove_accents else lines_str
    #
    #     tokenized = self.tokenizer.tokenize(lines.lower() if lower else lines)
    #
    #     return tokenized if split else " ".join(tokenized)

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



# import pickle
# from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktTrainer
# class PunctTokenizer:
#     def __init__(self):
#         with open( os.path.join(os.path.dirname(os.path.realpath(__file__)), "tokenizer_rus.pkl") , "rb") as tok_dump:
#             trainer = pickle.load(tok_dump)
#             tokenizer = PunktSentenceTokenizer(trainer.get_params())
#             tokenizer._params.abbrev_types.add('см')
#             tokenizer._params.abbrev_types |= set(["рр","гг","г","в","вв","м","мм","см","дм","л","км","га","кг","т","г","мг","бульв","г","д","доп","др","е","зам","Зам","и","им","инд","исп","Исп","англ","в","вв","га","гг","гл","гос","грн","дм","долл","е","ед","к","кап","кав","кв","кл","кол","комн","куб","л","лиц","лл","м","макс","кг","км","коп","л","лл","м","мг","мин","мл","млн","Млн","млрд","Млрд","мм","н","наб","нач","неуд","нем","ном","о","обл","обр","общ","ок","ост","отл","п","пер","Пер","перераб","пл","пос","пр","пром","просп","Просп","проф","Проф","р","ред","Рис","рус","с","сб","св","См","см","сов","соч","соц","спец","ср","ст","стр","т","тел","Тел","тех","тов","тт","туп","руб","Руб","тыс","Тыс","трлн","уд","ул","уч","физ","х","хор","э","ч","чел","шт","экз","Й","Ц","У","К","Е","Н","Г","Ш","Щ","З","З","Х","Ъ","Ф","Ы","В","А","П","Р","О","Л","Д","Ж","Ж","Э","Я","Ч","С","М","И","Т","Ь","Б","Ю"])
#             tokenizer._params.abbrev_types -= set(["задокументированна","короткозаострённые","«константиновский»","трансцендентализма","генерал-капитанами","взаимосотрудничества","1,1-дифтор-2-бромэтан","корректируются","ингерманландия","«копенгагеном»","воздушно-десант","несвоевременной","металлопродукции","аукштейи-паняряй","тёмно-фиолетовые","не-администратора","лингвист-психолог","лактобактериям","бекасово-сорт","фанатическая»","миннезингеров","коннерсройте","муковисцидоз","казахстан.т.е","дистрибьюции","выстрела/мин","турбулентное","блокбастером","кильписъярви","intraservice","леверкузене","заморозился","магнитского","канюк-авгур","бразильянки","махабхараты","таможеннике","выродженным","мальчевским","канторович","лабораторн","баттерфилд","ландшафтом","вымирающим","«фонтанка»","запоріжжя»","«амазонка»","разгребать","котируется","неразъемная","«линфилдом»","преупреждён","«чугунками»","focus-verl","ширшинский","гольфистом","обьединять","военнослуж","бхактапура","залежались","брокен-боу","церингенов","переделают","либреттист","перегонкой","глумились","критикуйте","котлетами»","крейстагом","шарлоттенб","вишневски","деконская","тарановка","трехгорка","коллоредо","шумановка","позолочен","прасолову","розоватые","меркушева","«гол+пас»","башлачёва","разгрести","«нурорда»делдается","золочение","«гломмен»","«марокко»","эстетично","пироговцы","wallpaper","огоромное","рогозянка","березицы","кольпите","warships","«двойка»","«русины»","аравакам","обозного","даргинец","нужности","дерегуса","«фалкон»","шингарка","омонимии","монфокон","парнэяха","пафосом»","снытиной","шихуанди","«жирона»","огородом","хивинск","шан-хай","/рэдкал","потенца","рычажки","геттинг","бургибы","отвилей","огрешки","фатьму","девайс","бербер","чувичи","неволю","шонгуй","нерпой","ганнов","алумяэ","штанах","клоака","рыксой","шкяуне","оффтоп","виднее","спам»","узолы","уйта","бяка","джос","тюля","пёза","уля"])
#             self.puncttok = tokenizer
            

#     def __call__(self, text):
#         return self.puncttok.tokenize(text)

#     def pickle(self):
#         pickle.dump(self.puncttok, open("russian.pickle", "wb"))


from nltk.data import load

class Sentencizer:
    def __init__(self, lang="rus"):
        lang = lang.lower()
        if lang in {'rus', "ru", "russian", ''}:
            language = 'russian'
        elif lang in {'eng', 'en', 'english'}:
            language = 'english'
        else:
            raise NotImplementedError(f"Language is not supported: {lang}")

        self.tokenizer = load("tokenizers/punkt/{0}.pickle".format(language))

    def __call__(self, text):
        return self.tokenizer.tokenize(text)


if __name__ == "__main__":
    text_en = """<doc id="1300" url="https://en.wikipedia.org/wiki?curid=1300" title="Abalone">
Abalone

Abalone ( or ; via Spanish "", from the Rumsen language "aulón") is a common name for any of a group of small to very large sea snails, marine gastropod molluscs in the family Haliotidae.
Other common names are ear shells, sea ears, and muttonfish or muttonshells in Australia, ormer in Great Britain, perlemoen in South Africa, and in New Zealand.
S.W.A.T. M. D.
"""

    text_ru = """Агноёстици́зм (от др.-греч. ἄγνωστος — непознанный) — философская концепция, согласно которой мир непознаваем и люди не могут знать ничего достоверного о действительной сущности вещей; позиция религиозного агностицизма заключается в том, что люди не могут знать ничего достоверного о Боге (или богах)[1][2][3][4][5]. В общеупотребительном смысле (особенно в англоязычной литературе) агностицизм нередко смешивают с атеизмом, со скептицизмом в отношении религии вообще[1][2][4]. """

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