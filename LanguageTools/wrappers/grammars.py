class EnNpGrammar:
    grammar = r"""
        # NP:
        #     {<DET>?<NOUN|ADJ><NOUN|ADJ|'s|of|in|with|for|on|over|throughout>*<NOUN>}
        NP:
            {<DET>?<NOUN|ADJ|'s>*<NOUN>}
        NP_of:
            {<NP><of><NP>}
        NP_in:
            {<NP><in><NP>}
        NP_with:
            {<NP><with><NP>}
        NP_for:
            {<NP><for><NP>}
        NP_on:
        # maybe this rule should only work in subconcepts
            {<NP><on><NP>}
        NP_over:
            {<NP><over><NP>}
        NP_throughout:
            {<NP><throughout><NP>}
        """
        # TODO:
        # 1. add 's detection DONE
        # 2. handle a variant ’s DONE
        # 3. names do not seem to parse
        # 4. and NP does not process 100% of the time
        # 5. NP of NP
        # 6. Such thing, or no such thing are two antipatterns
        # 7. Incorporate numerals
        #       In addition to relatively young projects, a number of major exchanges have made their choice in favor of Malta, including Binance, OKEx, ZB.com, as well as such famous blockchain projects as TRON, Big One, Cubits, Bitpay and others.
        # 8. Antipatterns
        #       rid of NP

class RuNpGrammar:
    grammar = r"""
            NP:
                {<NOUN|ADJ>*<NOUN.*>}
            """