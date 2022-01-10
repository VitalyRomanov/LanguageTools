from collections import deque
from pathlib import Path

from LanguageTools import Tokenizer
from LanguageTools.data.WikiLoader import WikiDataLoader
import argparse
import json

from LanguageTools.utils import MapReduce


tok = Tokenizer()


def map_fn(doc):
    return [(t.text, 1) for t in tok.token_gen(doc)]


def reduce_fn(left, right):
    if left is None:
        return right
    return left + right


def main():
    parser = argparse.ArgumentParser("Iterate wikipedia stored in JSON format")
    parser.add_argument("wiki_location")
    parser.add_argument("output")
    args = parser.parse_args()

    wiki = WikiDataLoader(args.wiki_location)
    # wiki = open(args.wiki_location, "r")

    def get_data():
        for doc in wiki:
            doc = json.loads(doc)['text']
            yield doc

    output = Path(args.output)

    mr = MapReduce(output, map_fn, reduce_fn)
    results = mr.run(get_data(), allow_parallel=True, desc="Processing wiki")

    with open(output.joinpath("wc"), "w") as sink:
        for key, value in results:
            sink.write(f"{key}\t{value}\n")


if __name__ == "__main__":
    main()

