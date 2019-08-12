import sys
from collections import Counter

# vocab_path = sys.argv[1]
# output_path = sys.argv[1]

normalized_words = Counter()

def normalize(word):
    return word.lower()

# with open(output_path, "w") as normalizaiton_edges:

# for line in open(vocab_path, "r"):
for line in sys.stdin:
    if line.strip():
        word, count_str = line.strip().split()
        count = int(count_str)

        normalized_word = normalize(word)

        # if word in normalized_words:
        #     normalized_words[normalized_word] += count
        # else:
        #     normalized_words[normalized_word] = count

        print("%s\t%s\tnormal_form" % (word, normalized_word))