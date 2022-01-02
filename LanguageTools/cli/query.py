import argparse
from time import time

from tqdm import tqdm

from LanguageTools.corpus import DocumentCorpus
from LanguageTools.rankers import BinaryRetrieval
from LanguageTools.rankers.BinaryRetrieval import BinaryRetrievalBiword


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus_path")
    args = parser.parse_args()

    ranker = BinaryRetrieval.load(args.corpus_path)
    ranker.add_bigrams = True

    while True:
        query = input("Enter query:")
        start = time()
        results = ranker.query(q=query.strip(), exact=True)
        end = time()
        print(f"Results in {end - start}, s: ", results)


if __name__ == "__main__":
    main()

