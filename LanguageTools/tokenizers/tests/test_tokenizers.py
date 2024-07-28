def test_tokenizer():
    from LanguageTools.tokenizers import Tokenizer

    text = '\rThis is usually how I do it.\tНо лучше не повторять.\n   Ciao.'
    nlp = Tokenizer("regexp")
    doc = nlp(text)
    assert (
        [tok.text for tok in doc.tokens] ==
        ['This', 'is', 'usually', 'how', 'I', 'do', 'it', '.', 'Но', 'лучше', 'не', 'повторять', '.', 'Ciao', '.']
    )

    text = '\rThis is usually how I do it.\tНо лучше не повторять.\n   Ciao.'
    nlp = Tokenizer("bert-base-uncased")
    doc = nlp(text)
    assert (
            [tok.text for tok in doc.tokens] ==
            ['this', 'is', 'usually', 'how', 'i', 'do', 'it', '.', 'н', '##о', 'л', '##у', '##ч', '##ш', '##е', 'н', '##е', 'п', '##ов', '##т', '##о', '##р', '##я', '##т', '##ь', '.', 'cia', '##o', '.']
    )

    text = '\rThis is usually how I do it.\tНо лучше не повторять.\n   Ciao.'
    nlp = Tokenizer("gpt2")
    doc = nlp(text)
    assert (
            [tok.text for tok in doc.tokens] ==
            ['č', 'This', 'Ġis', 'Ġusually', 'Ġhow', 'ĠI', 'Ġdo', 'Ġit', '.', 'ĉ', 'Ð', 'Ŀ', 'Ð¾', 'ĠÐ', '»', 'Ñĥ', 'Ñ', 'ĩ', 'Ñ', 'Ī', 'Ðµ', 'ĠÐ', '½', 'Ðµ', 'ĠÐ', '¿', 'Ð¾Ð', '²', 'ÑĤ', 'Ð¾', 'ÑĢ', 'Ñı', 'ÑĤ', 'ÑĮ', '.', 'Ċ', 'Ġ', 'Ġ', 'ĠC', 'iao', '.']
    )


def test_biluo():
    from LanguageTools.tokenizers import Tokenizer
    from LanguageTools.tokenizers import biluo_tags_from_offsets

    text = '\rThis is usually how I do it.\tНо лучше не повторять.\n   Ciao.'

    offsets = [[9, 25, "ent1"], [30, 32, "ent2"], [42, 51, "ent3"]]
    nlp = Tokenizer()
    doc = nlp(text)
    biluo = biluo_tags_from_offsets(doc, offsets)
    assert (
        biluo ==
        ['O', 'O', 'B-ent1', 'I-ent1', 'I-ent1', 'L-ent1', 'O', 'O', 'U-ent2', 'O', 'O', 'U-ent3', 'O', 'O', 'O']
    )


def test_alignment():
    def verify_tokens(text, tokens, spans, do_assert=True):
        last_token_end = None
        for ind, (t, sp) in enumerate(zip(tokens, spans)):
            expected_token = t.strip("▁").strip("#").replace("č", "\r").replace("ĉ", "\t").replace("Ċ", "\n")
            observed_token = text[sp[0]: sp[1]]
            if do_assert:
                assert expected_token == observed_token
            if ind > 0:
                observed_space = text[last_token_end: sp[0]]
            else:
                observed_space = ""
            last_token_end = sp[1]
            # print(repr(expected_token), repr(observed_token), repr(observed_space))
        # print()
    from LanguageTools.tokenizers import SubwordTokenizer

    text = '\rThis is usually how I do it.\tНо лучше не повторять.\n   Ciao.'

    tokens = ['▁', '\rT', 'his', '▁is', '▁usually', '▁how', '▁', 'I', '▁do', '▁it', '.', '\tН', 'о', '▁лучше', '▁не', '▁повторя', 'ть', '.', '\n', '▁', 'C', 'iao', '.']
    spans = SubwordTokenizer._align_surface(
        tokens=tokens,
        text=text
    )
    verify_tokens(text, tokens, spans)

    tokens = ['This', 'is', 'usually', 'how', 'I', 'do', 'it', '.', 'Н', '##о', 'л', '##у', '##ч', '##ш', '##е', 'н', '##е', 'п', '##ов', '##т', '##о', '##р', '##я', '##т', '##ь', '.', 'C', '##ia', '##o', '.']
    spans = SubwordTokenizer._align_surface(
        tokens=tokens,
        text=text
    )
    verify_tokens(text, tokens, spans)

    tokens = ['č', 'This', 'Ġis', 'Ġusually', 'Ġhow', 'ĠI', 'Ġdo', 'Ġit', '.', 'ĉ', 'Ð', 'Ŀ', 'Ð¾', 'ĠÐ', '»', 'Ñĥ', 'Ñ', 'ĩ', 'Ñ', 'Ī', 'Ðµ', 'ĠÐ', '½', 'Ðµ', 'ĠÐ', '¿', 'Ð¾Ð', '²', 'ÑĤ', 'Ð¾', 'ÑĢ', 'Ñı', 'ÑĤ', 'ÑĮ', '.', 'Ċ', 'Ġ', 'Ġ', 'ĠC', 'iao', '.']
    spans = SubwordTokenizer._align_bpe(
        tokens=tokens,
        text=text
    )
    verify_tokens(text, tokens, spans, do_assert=False)


# noinspection DuplicatedCode
def test_offset_extraction():
    from LanguageTools.tokenizers import SubwordTokenizer
    from LanguageTools.tokenizers.primitives import extract_spans

    text = '\rThis is usually how I do it.\tНо лучше не повторять.\n   Ciao.'

    offsets = [[9, 25], [29, 32], [42, 51]]
    tokenizer = SubwordTokenizer("bpemb")
    tokens = list(tokenizer.token_gen(text))
    token_spans = extract_spans(tokens, offsets)
    for token_span, char_span in zip(token_spans, offsets):
        span_tokens = tokens[token_span[0]: token_span[1]]
        observed_span_str = tokenizer.convert_tokens_to_string(span_tokens)
        expected_span_str = text[char_span[0]: char_span[1]]
        assert observed_span_str == expected_span_str, f"{observed_span_str} != {expected_span_str}"

    offsets = [[9, 25], [30, 32], [42, 51]]
    tokenizer = SubwordTokenizer("bert-base-cased", tokenizer_class="BertTokenizer")
    tokens = list(tokenizer.token_gen(text))
    token_spans = extract_spans(tokens, offsets)
    for token_span, char_span in zip(token_spans, offsets):
        span_tokens = tokens[token_span[0]: token_span[1]]
        observed_span_str = tokenizer.convert_tokens_to_string(span_tokens)
        expected_span_str = text[char_span[0]: char_span[1]]
        assert observed_span_str == expected_span_str, f"{observed_span_str} != {expected_span_str}"

    offsets = [[9, 25], [30, 32], [42, 51]]
    tokenizer = SubwordTokenizer("gpt2")
    tokens = list(tokenizer.token_gen(text))
    token_spans = extract_spans(tokens, offsets)
    for token_span, char_span in zip(token_spans, offsets):
        span_tokens = tokens[token_span[0]: token_span[1]]
        observed_span_str = tokenizer.convert_tokens_to_string(span_tokens)
        expected_span_str = text[char_span[0]: char_span[1]]
        assert observed_span_str == expected_span_str


# noinspection PyUnusedLocal
def test_bpe_tokenizer():
    from LanguageTools.tokenizers import SubwordTokenizer

    text = '\rThis is usually how I do it.\tНо лучше не повторять.\n   Ciao.'
    tokenizer = SubwordTokenizer("bpemb")
    tokens = list(tokenizer.token_gen(text))

    tokenizer = SubwordTokenizer("bert-base-cased", tokenizer_class="BertTokenizer")
    tokens = list(tokenizer.token_gen(text))

    tokenizer = SubwordTokenizer("gpt2")
    tokens = list(tokenizer.token_gen(text))
