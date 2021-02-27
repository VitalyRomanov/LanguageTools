import os
import sys


class WikiDataLoader:
    def __init__(self, path):
        # Path to extracted wiki dump
        self.path = path
        self.sub_path = None

        # List of documents from wiki dump file
        self.docs = []
        self.folders = []
        self.files = []

        # Prepare the list of subfolders in extracted wiki dump
        self.set_folders()

        # len(self)

    def set_folders(self):
        self.folders = list(filter(lambda x: os.path.isdir(os.path.join(self.path, x)), os.listdir(self.path)))
        self.folders.sort()

    def next_folder(self):

        if self.folders:
            c_folder = self.folders.pop(0)
            sub_path = os.path.join(self.path, c_folder)
            self.files = list(filter(lambda x: os.path.isfile(os.path.join(sub_path, x)), os.listdir(sub_path)))
            self.files = list(filter(lambda x: x[0] != '.', self.files))
            self.files.sort()

            self.sub_path = sub_path
        else:
            self.set_folders()

    def next_file(self):

        if not self.files:
            self.next_folder()

        if self.files:
            c_file = self.files.pop(0)
            c_path = os.path.join(self.sub_path, c_file)

            docs = open(c_path, "r").read().strip().split("\n")

            self.docs = docs

    def next_doc(self):
        """
        Return next available document
        """

        if not self.docs:
            self.next_file()

        if self.docs:
            return self.docs.pop(0)
        else:
            return None

    def __iter__(self):
        return self

    def __next__(self):
        doc = self.next_doc()
        if doc:
            return doc
        else:
            raise StopIteration()

    # def __len__(self):
    #     if not hasattr(self, "length"):
    #         n = 0
    #         for _ in self:
    #             n += 1
    #         self.length = n
    #     return self.length


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser("Iterate wikipedia stored in JSON format")
    parser.add_argument("wiki_location")
    args = parser.parse_args()

    wiki = WikiDataLoader(args.wiki_location)
    try:
        for doc in wiki:
            doc = json.loads(doc)['text']
            print(doc)
    except BrokenPipeError:
        sys.exit()
