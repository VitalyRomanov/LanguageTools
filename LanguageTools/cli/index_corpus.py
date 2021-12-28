import argparse

from LanguageTools.corpus import DocumentCorpus
from LanguageTools.rankers import BinaryRetrieval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path")
    parser.add_argument("corpus_path")
    args = parser.parse_args()

    corpus = DocumentCorpus(args.corpus_path, "en")

    with open(args.data_path, "r") as data:
        for line in data:
            corpus.add_docs([line], save_instantly=False)

    corpus.save()

    ranker = BinaryRetrieval(corpus, index_instantly=True)
    ranker.save()


if __name__ == "__main__":
    main()

