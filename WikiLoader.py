import os

class WikiDataLoader:
    def __init__(self, path):
        # Total number of lines, currently not used
        # self.line_count = 0
        # Current line count, not used 
        # self.current_line = None

        # Path to extracted wiki dump
        self.path = path
        # current file object in extracted wiki dump
        self.current_file = None
        # Count of current subfolder in extracted wiki dump
        self.subfolder_pos = -1
        # Count of the current file in the current subfolder 
        self.file_pos = 0

        # Prepare the list of subfolders in extracted wiki dump
        self.subfolders = list(filter(lambda x: os.path.isdir(os.path.join(path,x)), os.listdir(path)))
        self.subfolders.sort() # Ensure alphabetical order
        # List of documents from wiki dump file
        self.docs = None
        # Count of the current file in current 
        self.current_doc = -1
        # List of files in subfolder
        self.files = []

    def next_file(self):
        # Increment file position
        self.file_pos += 1
        
        # If file position exceeds number of files in current folder, change folder
        if self.file_pos >= len(self.files):
            self.subfolder_pos += 1
            if self.subfolder_pos > len(self.subfolders):
                # If all folders are visited, return None
                self.current_file = None
                return None
            sub_path = os.path.join(self.path, self.subfolders[self.subfolder_pos])
            self.files = list(filter(lambda x: os.path.isfile(os.path.join(sub_path,x)), os.listdir(sub_path)))
            self.files.sort()
            self.file_pos = 0

        # path to the current file
        file_path = os.path.join(os.path.join(self.path, self.subfolders[self.subfolder_pos]),self.files[self.file_pos])
        
        # If a file was opened, close it
        if self.current_file is not None:
            self.current_file.close()
        
        self.current_file = open(file_path, "r")


    def load_new_docs(self):
        self.next_file()

        # When no more files available, return None
        if self.current_file is None:
            self.docs = None

        docs = self.current_file.read().split("</doc>")
        docs_list = []

        for doc in docs:
            if doc.strip():
                # filter the first line that contains <doc> tag
                docs_list.append("\n".join(doc.split("\n")[1:]))

        self.docs = docs_list
        self.current_doc = 0

    def next_doc(self):
        '''
        Return next available document
        '''
        if self.current_file is None:
            self.next_file()

        self.current_doc += 1

        if self.docs is None or self.current_doc >= len(self.docs):
            self.load_new_docs()
            if self.current_doc is None:
                return None

        return self.docs[self.current_doc]


        # line = self.current_file.readline()
        # while line == "":
        #     self.next_file()
        #     if self.current_file is None:
        #         return False
        #     line = self.current_file.readline()
        # self.current_line = line.strip()
        # return True
#
#
# pairs = 0
# vocabulary = Counter()
#
# poss = {'ADJ','VERB','ADV','ADP','NOUN','PROPN','PRON','CONJ','DET','CCONJ','PART'}
#
# # files = {pos: open("%s_PAIRS" % pos, "w") for pos in poss}
#
# with open("wiki_pairs", "a") as pair_sets:
#     with open("dataset", "r") as dataset:
#         # line = dataset.readline()
#         wiki_data = WikiDataLoader("/home/ltv/data/wikipedia/en_wiki_plain")
#
#         # line = u"bright red apples on the tree. What's happened to me? he thought. It wasn't a dream."
#         # while line != "":
#         while wiki_data.next_line():
#             line = wiki_data.current_line
#             # print(line)
#             # continue
#             if len(line) > 0 and line[0] != "<":
#                 doc = nlp(line)
#
#                 for sent in doc.sents:
#                     for token in sent:
#                         token_code = token.text.lower() + "_" + token.pos_
#                         if token_code in vocabulary:
#                             vocabulary[token_code] += 1
#                         else:
#                             vocabulary[token_code] = 1
#
#                         if token.head == token:
#                             continue
#
#                         if token.pos_ in poss:
#
#                             pair_sets.write("%s %s\n" % (token_code, token.head.text.lower()+"_"+token.head.pos_))
#
#                             pairs += 1
#                             if pairs % 1000 == 0:
#                                 print("Processed %d pairs\r" % pairs, end="")
#
#             line = dataset.readline()
#
# print("Total number of pairs: %d" % pairs)
#
# pickle.dump(vocabulary, open("wiki_en_vocabulary", "wb"), protocol=4)
#
# # for token, count in vocabulary.most_common(100):
# #     print(token, count)
