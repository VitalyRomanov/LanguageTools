#!/usr/bin/python3
#By Steve Hanov, 2011. Released to the public domain
import time
import sys

# DICTIONARY = "downsampled_vocabulary.txt";
# TARGET = sys.argv[1]
# MAX_COST = int(sys.argv[2])


class TrieNode:
    # The Trie data structure keeps a set of words, organized with one node for
    # each letter. Each node has a branch for each letter that may follow it in the
    # set of words.
    def __init__(self, dawg):
        self.word = None
        self.children = {}
        
        self.dawg = dawg
        
        self.dawg.NodeCount += 1

    def insert( self, word ):
        node = self
        for letter in word:
            if letter not in node.children:
                node.children[letter] = TrieNode(self.dawg)

            node = node.children[letter]

        node.word = word


class DawgDictionary:
    def __init__(self, words=None):
        self.WordCount = 0
        self.NodeCount = 0
        
        self.trie = TrieNode(self)
        
        if words is None:
            pass
        else:
            self.add_words(words)
                
    def add_words(self, words):
        for word in words:
            self.add_word(word)

    def add_word(self, word):
        self.WordCount += 1
        self.trie.insert( word )
        
    def search(self, word, maxCost ):
        # The search function returns a list of all words that are less than the given
        # maximum distance from the target word
        
        # build first row
        currentRow = range( len(word) + 1 )

        results = []

        # recursively search each branch of the trie
        for letter in self.trie.children:
            self.searchRecursive( self.trie.children[letter], letter, word, currentRow,
                results, maxCost )

        return results
        
    def searchRecursive(self, node, letter, word, previousRow, results, maxCost ):
        # This recursive helper is used by the search function above. It assumes that
        # the previousRow has been filled in already.

        columns = len( word ) + 1
        currentRow = [ previousRow[0] + 1 ]

        # Build one row for the letter, with a column for each letter in the target
        # word, plus one for the empty string at column 0
        for column in range( 1, columns ):

            insertCost = currentRow[column - 1] + 1
            deleteCost = previousRow[column] + 1

            if word[column - 1] != letter:
                replaceCost = previousRow[ column - 1 ] + 1
            else:
                replaceCost = previousRow[ column - 1 ]

            currentRow.append( min( insertCost, deleteCost, replaceCost ) )

        # if the last entry in the row indicates the optimal cost is less than the
        # maximum cost, and there is a word in this trie node, then add it.
        if currentRow[-1] <= maxCost and node.word != None:
            results.append( (node.word, currentRow[-1] ) )

        # if any entries in the row are less than the maximum cost, then
        # recursively search each branch of the trie
        if min( currentRow ) <= maxCost:
            for letter in node.children:
                self.searchRecursive( node.children[letter], letter, word, currentRow,
                    results, maxCost )


def find_pairs(word, dawg, max_cost):
    pairs = []
    
    results = dawg.search( word, max_cost )
    
    for r in results:
        if r[1] > 0:
            pair = [word, r[0]]
            pair.sort()
            pair_t = tuple(pair)
            pairs.append((pair_t, r[1]))
    
    return pairs


def compute_allowed_cost(word_len):
    if word_len < 5:
        allowed_cost = 1
    elif word_len < 8:
        allowed_cost = 2
    else:
        allowed_cost = 3
    return allowed_cost
    

if __name__ == "__main__":
    import sys#, argparse
    
    # parser = argparse.ArgumentParser(description='Process some integers.')
    # parser.add_argument('word_list', metavar='W', type=str,
    #                     help='file with list of files')
    # parser.add_argument('-m', dest='mode', type=str,
    #                     help='depending on the mode stream/file reads and output words from stream or from the same file as dictionary')
    # parser.add_argument('-o', dest='output', type=str,
    #                     help='destination to write word pairs in the mode \'file\'')
                        
    # args = parser.parse_args()
    
    # if args.mode == "stream":
    #     pass    
    # elif args.mode == "file":
    #     try:
    #         out_sink = args.output
    #     except:
    #         print("Specify output destination for the mode \'file\'")
    #         sys.exit()
    # else:
    #     raise Exception("Unsupported mode. Choose \'stream\' or \'file\'")

    seen = set()        
    MAX_COST = 1
    
        
    # words = open(args.word_list, "rt").read().strip().split()
    # dawg = DawgDictionary(words)
    
    
    # if args.mode == "stream":
    #     for line in sys.stdin:
    #         words = line.strip().split()
    #         for ind, word in enumerate(words):
    #             pairs = filter(lambda pair: pair not in seen, find_pairs(word, dawg, MAX_COST))
    #             seen.update(pairs)
    #             print(pairs)
    #             for pair in pairs:
    #                 print("%s %s" % (pair[0], pair[1]))
    
    # elif args.mode == "file":
    #     with open(args.output, "w") as pairs_sink:
    #         for ind, word in enumerate(words):
    #             pairs = filter(lambda pair: pair not in seen, find_pairs(word, dawg, MAX_COST))
    #             seen.update(pairs)
    #             for pair in pairs:
    #                 pairs_sink.write("%s %s\n" % (pair[0], pair[1]))
                
    #                 print("\rProcessed %d/%d" % (ind+1, len(words)))

    #         print("%d pairs found" % len(seen))

    phonetic_map_path = sys.argv[1]
    max_cost = int(sys.argv[2])
    output_path = sys.argv[3]

    all_words = []
    for line in open(phonetic_map_path, "r"):
        if line.strip():
            try:
                normal_word, phonetic, relationship = line.strip().split("\t")
            except:
                print("Error: ", line)
                sys.exit()
            if phonetic: #phonetic can be empty in some mappings
                all_words.append(phonetic)

    all_words.sort()

    dawg = DawgDictionary(all_words)

    with open(output_path, "w") as sink:
        for ind, word in enumerate(all_words):

            allowed_cost = min(compute_allowed_cost(len(word)), max_cost)

            pairs_with_cost = find_pairs(word, dawg, allowed_cost)

            for pair, cost in pairs_with_cost:
                # pairs are sorted by alphabet
                if pair in seen: continue
                sink.write("%s\t%s\tphonetic_sim_%d\n" % (pair[0], pair[1], cost))
                seen.add(pair)

            print("\r%d/%d" % (ind, len(all_words)), end="")
            


    




