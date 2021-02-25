import sys
from LanguageTools.Tokenizer import Sentencizer
from LanguageTools.corpus.DocumentCorpus import DocumentCorpus
from LanguageTools.corpus.TokenizedCorpus import TokenizedCorpus

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("lang")
    parser.add_argument("target_location")
    parser.add_argument("-c", "--corpus_type", dest="corpus_type", default="TokenizedCorpus", help="TokenizedCorpus|DocumentCorpus")
    args = parser.parse_args()

    if args.corpus_type == "TokenizedCorpus":
        sentencizer = Sentencizer(args.lang)
        corpus = TokenizedCorpus(args.target_location)

        def preprocess(text):
            return (s for s in sentencizer(text))

    elif args.corpus_type == "DocumentCorpus":
        corpus = DocumentCorpus(args.target_location, args.lang)

        def preprocess(text):
            return [text]
    else:
        raise ValueError()

    for line in sys.stdin:
        corpus.add_docs(preprocess(line), save_instantly=False)

    corpus.save()