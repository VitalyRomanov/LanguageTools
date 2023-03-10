import unicodedata


def deaccent(text):
    """
    Remove accentuation from the given string. Input text is either a unicode string or utf8 encoded bytestring.

    Return input string with accents removed, as unicode.

    >>> deaccent("Šéf chomutovských komunistů dostal poštou bílý prášek")
    u'Sef chomutovskych komunistu dostal postou bily prasek'

    """
    if not isinstance(text, str):
        # assume utf8 for byte strings, use default (strict) error handling
        text = text.decode('utf8')
    norm = unicodedata.normalize("NFD", text)
    result = "".join(ch for ch in norm if unicodedata.category(ch) != 'Mn')
    return unicodedata.normalize("NFC", result)


def test_deaccent():
    text = """
    aulón"
    Агноёстици́зм (от др.-греч. ἄγνωστος — непознанный).
    """
    deacc = deaccent(text)
    assert deacc == '\n    aulon"\n    Агноестицизм (от др.-греч. αγνωστος — непознанныи).\n    '


if __name__ == "__main__":
    test_deaccent()