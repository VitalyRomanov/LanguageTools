from collections import Counter
from math import log, exp

# Normalized (Pointwise) Mutual Information in Collocation Extraction

wiki_sent_tok_id_file = "/Volumes/Seagate/language data/ru_wikipedia/articles/wiki_sent_tok_id"
# wiki_sent_tok_id_gram_file = "/Volumes/Seagate/language data/ru_wikipedia/articles/wiki_sent_tok_id_gram"

threshold = 0.8
margin = 0

def make_inv_voc(vocab):
    return {id: word for word, id in vocab.items()}

def load_voc():
    vocab = dict()
    with open("wiki_vocab", "r") as vocab_file:
        lines = vocab_file.read().split("\n")
        for line in lines:
            elem = line.split(" ")
            if len(elem) == 2:
                vocab[elem[0]] = int(elem[1])

        return vocab

def load_counts(vocab):
    token_counter = Counter()
    with open("wiki_tokens", "r") as token_file:
        lines = token_file.read().split("\n")
        for line in lines:
            elem = line.split(" ")
            if len(elem) == 2:
                token_counter[vocab[elem[0]]] = int(elem[1])
        return token_counter

print("Loading vocabulary")
vocab = load_voc()
inv_vocab = make_inv_voc(vocab)
print("Loading token counts")
counts = load_counts(vocab)

extension_mapping = dict()
source = wiki_sent_tok_id_file
target = source + "_"

for _ in range(3):

    bigram_voc = Counter()

    with open(source, "r") as data:
        line = data.readline()

        proc_lines = 1

        while line != "":
            ids_str = list(filter(lambda x: x != "", line.strip().split(" ")))
            ids = list(map(lambda x: int(x), ids_str))

            for i in range(len(ids)-1):
                if ids[i] > margin or ids[i+1] > margin:
                    bgram = (ids[i], ids[i+1])

                    if bgram in bigram_voc:
                        bigram_voc[bgram] += 1
                    else:
                        bigram_voc[bgram] = 1

            line = data.readline()
            proc_lines += 1
            if proc_lines % 1000 == 0:
                print("\rProcessed: %d, bigrams: %d" % (proc_lines, len(bigram_voc)), end='')
            # if proc_lines == 1000000: break

    print("")
    margin = len(vocab)

    if len(bigram_voc) == 0:
        break

    total_words = log(sum(counts.values()))
    total_bigrams = log(sum(bigram_voc.values()))

    for bigram, count in bigram_voc.items():
        pwmi = (log(count) - log(counts[bigram[0]]) - log(counts[bigram[1]]) + \
                2 * total_words - total_bigrams) / (total_bigrams - log(count))
        if pwmi < threshold:
            bigram_voc[bigram] = -1

    bigram_voc = +bigram_voc

    def extend_vocab(vocab, counts, ngramms):
        extension_mapping = dict()
        for gram in ngramms:
            gram_string = "%s %s" % (inv_vocab[gram[0]], inv_vocab[gram[1]])
            new_id = len(vocab)
            vocab[gram_string] = new_id
            extension_mapping[gram] = new_id
            counts[new_id] = ngramms[gram]
        return extension_mapping

    extension_mapping = extend_vocab(vocab, counts, bigram_voc)
    inv_vocab = make_inv_voc(vocab)

    with open("wiki_bigram", "a") as wb:
        for bigram, count in bigram_voc.most_common(len(bigram_voc)):
            pwmi = (log(count) - log(counts[bigram[0]]) - log(counts[bigram[1]]) + \
                    2 * total_words - total_bigrams) / (total_bigrams - log(count))
            wb.write("%s %s %d %f\n" % (inv_vocab[bigram[0]], inv_vocab[bigram[1]], count, pwmi))

    with open(target, "w") as gram_wiki:
        with open(source, "r") as wiki:
            sent_count = 0
            line = wiki.readline()
            while line!="":
                ids_str = list(filter(lambda x: x != "", line.strip().split(" ")))
                ids = list(map(lambda x: int(x), ids_str))

                i = 0
                new_ids = []
                while i < len(ids)-1:
                    bgram = (ids[i], ids[i+1])

                    if len(ids) > 1 and bgram in extension_mapping:
                        new_ids.append(extension_mapping[bgram])
                        # print("Substiture")
                        i += 2
                    else:
                        new_ids.append(ids[i])
                        i += 1

                for t in new_ids:
                    gram_wiki.write("%d " % t)
                gram_wiki.write("\n")
                sent_count += 1
                # if sent_count == 1000000: break
                line = wiki.readline()

    source = target
    target = source + "_"
    threshold += .03
    print("Finished round")

with open("wiki_vocab_grams","w") as wiki_vocab_dump:
    for token, id in vocab.items():
        wiki_vocab_dump.write("%s %d\n" % (token, id))

with open("extension_mapping","w") as e_dump:
    for gram, id in extension_mapping.items():
        e_dump.write("%d %d %d\n" % (gram[0], gram[1], id))
