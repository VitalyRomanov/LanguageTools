import sys
from collections import defaultdict

from conllu import parse


def read_conll(path):
    return parse(open(path, "r").read())


def get_conll_spaces(conll):
    sents = [[]]
    labels = [[]]

    for sent in conll:
        for token in sent:
            token_string = token["form"]
            token_space_after = True
            if token["misc"] is not None:
                misc = token["misc"]
                if "SpaceAfter" in misc:
                    token_space_after = False if misc["SpaceAfter"] == "No" else True

            sents[-1].append(token_string)
            labels[-1].append(token_space_after)
        sents.append([])
        labels.append([])

    if len(sents[-1]) == 0:
        sents.pop(-1)
        labels.pop(-1)

    return sents, labels



# def read_conll(path):
#     def empty_sent(fields=None):
#         return defaultdict(list)
#
#     # assert "token" in fields, "One of the fields should represent `token`"
#     sentences =
#
#     sents = [empty_sent(fields)]
#     current = sents[-1]
#     with open(path, "r") as conll:
#         for ind, line in enumerate(conll):
#             line = line.strip()
#             if line == "":
#                 sents.append(empty_sent(fields))
#                 current = sents[-1]
#
#             parts = line.split()
#             assert len(parts) == len(fields), \
#                 f"Error reading line {ind+1}, requesed size does not match actual data: {parts} != {fields}"
#             for p, f in zip(parts, fields):
#                 current[f].append(p)
#
#     def assign_spaces(tokens):
#         no_space_before = {",", ".", "!", "?"}
#         for ind, token in enumerate(tokens):
#             if ind == 0:
#                 pass
#             else:
#                 pass
#
#
#     return sents


if __name__ == "__main__":
    get_conll_spaces(read_conll(sys.argv[1]))