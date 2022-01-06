import argparse

from tqdm import tqdm

from LanguageTools.corpus import DocumentCorpus
from LanguageTools.rankers import BinaryRetrieval
from LanguageTools.rankers.BinaryRetrieval import BinaryRetrievalBiword


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("corpus_path")
    args = parser.parse_args()

    # corpus = DocumentCorpus(args.corpus_path, "en")
    corpus = DocumentCorpus.load(args.corpus_path)

    # with open(args.data_path, "r") as data:
    #     for line in tqdm(data):
    #         corpus.add_docs([line], save_instantly=False)
    #
    # corpus.save()

    ranker = BinaryRetrieval(corpus, index_instantly=True, add_bigrams=True)
    ranker.save()


if __name__ == "__main__":
    main()

