from collections import Counter

from vocabulary_loader import load_voc, load_counts, make_inv_voc

restricted = {'',',','!','"','#','$','%','&','\'','(',')','+','-','.','/','<','>','?','@','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','}','~','£','§','©','°','±','²','·','¹','¼','½','¾','×','à','ä','è','ö','š','ə','α','β','γ','δ','ε','ζ','η','θ','λ','μ','ο','π','ρ','σ','φ','χ','ω','в','г','д','е','з','й','л','м','н','п','р','т','ф','х','ц','ч','ш','щ','ъ','ы','ь','э','ю','і','ў','‒','–','—','―','„','†','•','…',' ','‰','€','№','™','←','↑','→','↔','−','≈','≤','≥','─','●'}

pairs_path = "../find_similar/pairs2.txt" 
voc_path = "../ru_wiki_processing_resources"

pairs = open(pairs_path, "r").read().split("\n")
pairs = list(map(lambda x: x.split(" "), pairs))

print("Loading vocabulary")
vocab = load_voc(voc_path)
inv_vocab = make_inv_voc(vocab)
print("Loading token counts")
counts = load_counts(voc_path, vocab)

print("%d pairs loaded" % len(pairs))

order = Counter()

with open("filtered_popular_pairs_include_prepositions.txt", "w") as fp:
    sorted_pairs = []

    for pair in pairs:
        the_pair = pair
        the_pair.sort()
        sorted_pairs.append(" ".join(the_pair))

    sorted_pairs.sort()

    for ind in range(0, len(sorted_pairs)):
        if sorted_pairs[ind] != sorted_pairs[ind-1]:
            words = sorted_pairs[ind].split(" ")
            if len(words) != 2 or words[0] in restricted or words[1] in restricted or counts[vocab[words[0]]] < 200 or counts[vocab[words[1]]] < 200:
                continue
            order[sorted_pairs[ind]] = max(counts[vocab[words[0]]],counts[vocab[words[1]]])

    for pair, count in order.most_common(len(order)):
        words = pair.split(" ")
        if len(words)!=2: continue
        fp.write("%s; %s; %d;\n" % (words[0], words[1], count))