from typing import List

from types import GeneratorType

from LanguageTools.utils.extra_utils import tokens_contain


def extract_spans(tokens, spans):
    spans = sorted(sorted(spans, key=lambda x: x[1]), key=lambda x: x[0])

    start = -1
    span_pos = 0
    curr_span = spans[span_pos]
    offset_label = None if len(curr_span) != 3 else curr_span[2]

    token_spans = []

    for ind, token in enumerate(tokens):
        cum_offset = token.start
        if curr_span[0] < cum_offset and start == -1:
            raise Exception("Offsets overlap or no alignment")

        if curr_span[0] == cum_offset and start == -1:
            start = ind

        cum_offset += token.length

        if cum_offset == curr_span[1]:
            token_spans.append((start, ind + 1, offset_label))
            start = -1
            span_pos += 1
            if span_pos == len(spans):
                break
            curr_span = spans[span_pos]
            offset_label = None if len(curr_span) != 3 else curr_span[2]

        cum_offset += token.tailspace

    return token_spans


def biluo_tags_from_offsets(doc, offsets):
    token_spans = extract_spans(doc, offsets)

    span_pos = 0
    current_span = token_spans[span_pos]
    span_ready = False

    biluo = []
    for ind, token in enumerate(doc):
        if current_span is None:
            prefix = "O"
        else:
            if ind == current_span[0]:
                if ind + 1 == current_span[1]:
                    prefix = "U-"
                    span_ready = True
                else:
                    prefix = "B-"
            elif ind + 1 == current_span[1]:
                prefix = "L-"
                span_ready = True
            elif current_span[0] < ind <= current_span[1]:
                prefix = "I-"
            else:
                prefix = "O"

        if prefix == "O":
            biluo.append("O")
        else:
            biluo.append(prefix + current_span[2])

        if span_ready is True:
            span_ready = False
            span_pos += 1
            current_span = token_spans[span_pos] if span_pos < len(token_spans) else None

    return biluo


class Token:
    id = None
    text = None
    tailspace = None
    start = None
    length = None

    def __init__(self, text, tailspace, id=None, **kwargs):

        self.id = id
        self.text = text
        self.tailspace = tailspace

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        return repr(self.text)

    def __eq__(self, other):
        if isinstance(other, int):
            if self.id is None:
                raise ValueError("Cannot compare using id, id for token is not assigned")
            return self.id == other
        elif isinstance(other, str):
            return self.text == other

    def len(self):
        if self.length is not None:
            return self.length
        elif self.text is not None:
            return len(self.text)
        else:
            raise ValueError(f"Token length is not available: {self}")


class Doc:
    def __init__(self, tokens, vocabulary=None):
        self.vocabulary = vocabulary
        if isinstance(tokens, list):
            self.tokens = tokens
        if isinstance(tokens, GeneratorType):
            self.tokens = list(tokens)

    def __iter__(self):
        return iter(self.tokens)

    def __len__(self):
        return len(self.tokens)

    def __repr__(self):
        return str(self)

    def __getitem__(self, item):
        return self.tokens[item]

    def __str__(self):
        str_ = ""
        for t in self.tokens:
            if t.text is None and self.vocabulary is not None:
                t.text = self.vocabulary[t.id]
            str_ += t.text
            if t.tailspace:
               str_ += " "
        return str_

    def __contains__(self, tokens: List[int]):
        return tokens_contain(self.tokens, tokens) == 1


class MultiDoc:
    def __init__(self, docs):
        if isinstance(docs, list):
            self.docs = docs
        if isinstance(docs, GeneratorType):
            self.docs = list(docs)

    def __iter__(self):
        return iter(self.docs)

    def __len__(self):
        return sum(len(doc) for doc in self.docs)

    def __repr__(self):
        return str(self)

    def __getitem__(self, item):
        return self.docs[item]

    def __str__(self):
        return " ".join(str(doc) for doc in self.docs)

    def __contains__(self, tokens: List[int]):
        for doc in self.docs:
            if tokens in doc:
                return True
        else:
            return False
