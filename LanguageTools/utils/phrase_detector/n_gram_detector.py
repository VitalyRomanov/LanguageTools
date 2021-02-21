from nltk import RegexpTokenizer, pos_tag
import pickle 
import argparse
from collections import Counter
from math import log


parser = argparse.ArgumentParser(description='Train word vectors')
parser.add_argument('input_file', type=str, default=150, help='Path to text file')
# parser.add_argument('output_dir', type=str, default=5, help='Ouput saving directory')
args = parser.parse_args()


tokenizer = RegexpTokenizer('[A-Za-zА-Яа-я-]+|[^\w\s]')

titles_articles = pickle.load(open(args.input_file, "rb"))
titles, articles = zip(*titles_articles)
articles = [tokenizer.tokenize(ta) for ta in articles]

def get_candidates(articles, threshold=0.5):
    """
    Creates a dictionary of candidate stable n-grams that can be merged into one token
    using npmi (normalized PMI)
    """
    regular_dict = Counter()
    bigram_dict = Counter()

    for article in articles:
        for ind, token in enumerate(article):
            # Calculate frequency of tekens
            if token in regular_dict:
                regular_dict[token] += 1
            else:
                regular_dict[token] = 1
            # And n-grams
            if ind + 1 < len(article):
                bigram = (article[ind], article[ind + 1])
                if bigram in bigram_dict:
                    bigram_dict[bigram] += 1
                else:
                    bigram_dict[bigram] = 1

    print("Unique tokens", len(regular_dict))
    print("Bigrams", len(bigram_dict))

    token_sum = sum(regular_dict.values())
    bigram_sum = sum(bigram_dict.values())
    candidates  = Counter()

    # pmi_counter = Counter()
    # Calculate npmi fofr every bigram
    for bigram in bigram_dict:
        p_x = regular_dict[bigram[0]] / token_sum
        p_y = regular_dict[bigram[1]] / token_sum
        p_x_y = bigram_dict[bigram] / bigram_sum

        pmi = log(p_x_y / (p_x * p_y))

        npmi = pmi / (- log(p_x_y))
        # pmi_counter[bigram] = pmi * p_x_y

        if npmi > threshold:
            # Allow only nouns and adjectives as stable phrases
            allowed = {'JJ', "NN", 'NNP', 'NNS'}
            tagged = pos_tag(bigram)
            if tagged[0][1] in allowed and tagged[1][1] in allowed:
                candidates[bigram] = npmi
                # candidates.add((bigram, npmi))
                # candidates.add(bigram)
                # print(bigram, npmi)
    return candidates

def collapse_candidates(articles, candidates):
    """
    Merge tokens into candidate stable n-grams.
    """
    collapsed_articles = []

    for article in articles:
        # merge tokens beginning from the back of the sentence
        article_rev = list(reversed(article))
        ind = 0
        while ind + 1 < len(article_rev):
            if (article_rev[ind + 1], article_rev[ind]) in candidates:
                new_token = article_rev[ind + 1] + "_" + article_rev[ind]
                # print(new_token)
                article_rev.pop(ind)
                article_rev.pop(ind)
                article_rev.insert(ind, new_token)
            ind += 1
        collapsed_articles.append(list(reversed(article_rev)))
    
    return collapsed_articles




candidates = get_candidates(articles, threshold=0.5)
articles = collapse_candidates(articles, candidates)
# print(articles[0])
candidates = get_candidates(articles, threshold=0.6)
articles = collapse_candidates(articles, candidates)
# print(articles[0])
candidates = get_candidates(articles, threshold=0.7)
articles = collapse_candidates(articles, candidates)

article_texts = [" ".join(article) for article in articles]

pickle.dump(list(zip(titles, article_texts)), open("articles_n_gram.pkl", "wb"))

# print(articles[0])

# regular_dict = Counter()
# for article in articles:
#     for ind, token in enumerate(article):
#         if token in regular_dict:
#             regular_dict[token] += 1
#         else:
#             regular_dict[token] = 1

# from pprint import pprint
# pprint(candidates)
# pprint(regular_dict.most_common(1500))

        

    


    