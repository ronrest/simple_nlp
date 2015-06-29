"""====================================================
                    DESCRIPTION

=======================================================
"""
__author__ = 'ronny'
from __future__ import print_function

import nltk

#TODO: Check that "maxent_treebank_pos_tagger" is installed, if not then:
#      nltk.download("maxent_treebank_pos_tagger")
#      this is required for POS tagging.

# ==============================================================================
#                                                                       TOKENIZE
# ==============================================================================
def tokenize(text, levels_out=1):
    """
    Takes a string of text, and returns a list of tokenized words.

    You can chose to have the tokenized words nested at different levels if you
    wish to group by sentences (levels_out=2), or paragraphs (levels_out=3).

    :param text: (string) The input string that you want to tokenize
    :param levels_out: (int) nesting level of the word tokens.
                       (1) if you want a single list of all word tokens.
                           (Default)
                       (2) if you want it split up into a list of
                           sentences,and those sentences are lists of word
                           tokens.
                       (3) if you want it split up into a list of paragraphs,
                           where paragraphs are lists of sentences, which are
                           lists of word tokens.
    :return: Depending on the value of levels used, it returns a list of
             strings,or a list of list of strings, or a list of list of list of
             strings.
    """
    assert isinstance(text, str), \
        "Argument *text* in tokenize() must be a string"
    assert isinstance(levels_out, int), \
        "Argument *levels_out* in tokenize() must be an integer"
    assert (levels_out >=1) and (levels_out <=3), \
        "Argument *levels_out* in tokenize() can only take the values 1, 2 or 3"

    if (levels_out == 1):
        return(nltk.word_tokenize(text))
    if (levels_out == 2):
        sentences = nltk.sent_tokenize(text)
        return([nltk.word_tokenize(sentence) for sentence in sentences])
    if (levels_out == 3):
        paragraphs = text.split("\n")
        tokenized = []
        for paragraph in paragraphs:
            sentences = nltk.sent_tokenize(paragraph)
            tokenized.append([nltk.word_tokenize(sentence) for sentence in sentences])
        return(tokenized)


# ==============================================================================
#                                                                    POS TAGGING
# ==============================================================================
# TODO: make it accept a single string that is not nested in a list.
def pos_tag(tokens):
    """
    Takes a list of tokens and returns a list of "Part of Speech" (POS) Tagged
    tupples.

    You feed in a list which contains tokens. The tokens can be nested between 1
    to 3 levels deep, and it will return the tupples at the same level of
    nesting.

    - if the token strings are at the top level (tokens[0] returns a string)
      then levels=1.
    - If you have tokens group by sentences (tokens[i][0] returns a string token)
      then levels_out=2.
    - If you have grouped by paragraphs (tokens[i][j][0] returns a string token)
      then levels_out=3.

    :param tokens: (list) The list of token strings
    :return: (list) Depending on how deep the nesting of tokens is, it returns
             a list of tuples, where the tuples are nested at the same level.

    :examples:
        # one level deep
        a = ["running", "for","their", "life"]

        # two levels deep
        b = [["running", "for","their", "life"],["Swimming", "towards", "sharks"]]
        c = [["hiking", "tall","mountains"],["Driving", "windy", "roads"]]

        # three levels deep
        d = [b, c]

        # handles all levels of nesting with same function call.
        pos_tag(a)
        pos_tag(b)
        pos_tag(d)
    """
    # ==========================================================================
    levels = get_level(tokens)
    return _multilevel_call(tokens,
                            list_type = "token",
                            level1_func = nltk.pos_tag,
                            level2_func = nltk.pos_tag_sents)


# ==============================================================================
#                                                                      GET LEVEL
# ==============================================================================
# TODO: test that all elements of the list are consistently the same depth.
def get_level(x, type="token", max_level=3):
    """
    Takes a list of token strings, and returns how many levels deep the tokens
    are.

    :param x: (list) The list of tokens
    :param type: (string) the type of element to look for.
                 type="token" looks for token strings
                 type="pos_tagged" looks for tuples with two string elements.
    :param max_level (int) maximum allowed nested depth.
    :return: (int) an integer representing how many levels deep the desired
              items are
    """
    # ==========================================================================
    assert isinstance(x, list), \
        "Argument *x* in get_level() function must be a list"
    assert (type == "token") or (type == "pos_tagged"), \
        "Argument *type* in get_level() function only accepts the values " \
        "'token' or 'pos_tagged'"


    if type == "token":
        data_type = str
        data_type_description = "strings"
    elif type == "pos_tagged":
        data_type = tuple
        data_type_description = "tuples"

    current_nest = x[0]
    level = 1
    while level <= max_level:
        if isinstance(current_nest, data_type):
            break
        else:
            assert isinstance(current_nest, list), \
                "Your {0} list must be a nested list of {1}"\
                "".format(type, data_type_description)
            current_nest = current_nest[0]
            level += 1
    if (level > max_level):
        #TODO: throw an error exception
        assert level <= max_level, \
             "You are in too deep! \nThe elements in your {0} list must be no "\
             "deeper than {1} levels deep.".format(type, max_level)
        return(None)
    else:
        return(level)


# ==============================================================================
#                                                          CHUNK PRESET PATTERNS
# ==============================================================================

# ------------------------------------------------------------------------------
# Noun Phrase Chunking Patterns
# ------------------------------------------------------------------------------

# This pattern was taken from
# https://github.com/lukewrites/NP_chunking_with_nltk
# Written by: lukewrites
CHUNK_PATTERN_NP1 = """
    NP: {<JJ>*<NN>+}
    {<JJ>*<NN><CC>*<NN>+}
"""

# This pattern was taken from
# https://github.com/lukewrites/NP_chunking_with_nltk
# Written by: lukewrites
CHUNK_PATTERN_NP2  = """
CHUNKED_NP:     {<DT><WP><VBP>*<RB>*<VBN><IN><NN>}
                {<NN|NNS|NNP|NNPS><IN>*<NN|NNS|NNP|NNPS>+}
                {<JJ>*<NN|NNS|NNP|NNPS><CC>*<NN|NNS|NNP|NNPS>+}
                {<JJ>*<NN|NNS|NNP|NNPS>+}
"""


# This pattern was taken from
# http://pythonprogramming.net/chunking-nltk-tutorial/
# Written by: Harrison Kinsley
CHUNK_PATTERN_NP3 = """CHUNKED_NP: {<RB.?>*<VB.?>*<NNP>+<NN>?}"""


# ==============================================================================
#                                                                          CHUNK
# ==============================================================================
def chunk(tagged_list, pattern=CHUNK_PATTERN_NP2, ne=False, binary_ne=False):
    """
    Takes a list of POS tagged items, and chunks them according to the pattern.

    Depending on how deeply nested the POS tagged items are, will determine
    how deeply nested the tree chunk are.

    Note, that in the pattern, not only can you specify the chunking regular
    expressions, but you can also specify the chinking regular expressions to
    remove things subtractively after chunking. For chinking, just use the
    curly brackets in reverse  eg:
         } patern_here {

    :param tagged_list: a list of POS tagged items.
    :param pattern: a chunking pattern. Several preset patterns exist:

                    - CHUNK_PATTERN_NP1  A simple pattern for Noun Phrases
                    - CHUNK_PATTERN_NP2  A better pattern for Noun Phrases
                    - CHUNK_PATTERN_NP3  Another pattern for Noun Phrases

    :param ne: (boolean) Make use of nltk's built in Named Entity Recognition?
    :param binary_ne: (boolean) Should the built in Named Entity Recognition use
                      binary classification?

                      - True - will classify the chunks as either Named Entities
                        or not.
                      - False - will classify the Named Entities into different
                        types, such as PERSON, ORGANIZATION, FACILITY, ... etc.

    :return: returns a list of trees, with certain tokens chunked together based
             on rules specified by the *pattern* variable.

    :examples:
        s = "Joe Blogs gave us tickets to the show.\n" \
            "The humorous comedian entertained Alice and Bob. "\
            "The clown, however, frightened us."
        t = tokenize(s, levels_out=3)
        pos_tagged=pos_tag(t)

        # Chunk based on Noun Phrases using the CHUNK_PATTERN_NP2 template
        noun_phrase_chunked = chunk(pos_tagged, pattern=CHUNK_PATTERN_NP2)
        noun_phrase_chunked[0][0].draw()
        noun_phrase_chunked[1][0].draw()

        # Use the built nltk's built in Named Entity Recognition for chunking
        ne_chunked = chunk(pos_tagged, ne=True)

        # As above, but categorize the named entities by type
        ne_chunked = chunk(pos_tagged, ne=True, binary_ne=False)

    """
    # ==========================================================================
    # Handle the use of the built in Named Entity Recognition
    if ne:
        return (named_entities(tagged_list, binary_ne))

    # Perform Chunking based on Regex Pattern
    levels = get_level(tagged_list, type="pos_tagged") # Depth of tagged_list
    chunker = nltk.RegexpParser(pattern)               # nltk regex object
    return _multilevel_call(tagged_list,
                                list_type = "pos_tagged",
                                level1_func = chunker.parse)


# ==============================================================================
#                                                                 NAMED ENTITIES
# ==============================================================================
def named_entities(tagged_list, binary):
    """
    Takes a list of nested POS tagged tuples, and chunks them using nltk's
    recomended named entity chunker.

    Depending on how deeply nested the POS tagged items are, will determine
    how deeply nested the returned trees are.

    You can specify that you want the named entities to be classified by
    different types of category by setting *binary* to False.

    :param tagged_list: (list) A nested list of POS tagged tuples
    :param binary: (boolean) should it use binary Named Entity classification?

                   - True  - Anything that is a named entity will be labelled NE
                   - False - Will classify the Named Entities into different
                             types,like PERSON, ORGANIZATION, FACILITY, ... etc.
    :return: returns a list of trees, with certain tokens chunked together.
    """
    # ==========================================================================
    levels = get_level(tagged_list, type="pos_tagged")  # Depth of tagged_list
    return _multilevel_call(tagged_list,
                            list_type="pos_tagged",
                            level1_func=nltk.ne_chunk,
                            level2_func=nltk.ne_chunk_sents,
                            binary=binary)


# ==============================================================================
#                                                                MULTILEVEL CALL
# ==============================================================================
def _multilevel_call(x, list_type, level1_func, level2_func=None, **kwargs):
    """
    A convenience function that takes a list of nested elements and a pointer
    to a function that processes the elements, and returns another nested list
    that contains the processed elements.

    Assumes that the first arguments of level1_func and level2_func are the
    list x (or its appropriate subsets depending on level)

    :param x: (list) The list contianing the elements
    :param list_type: The type of elements the list contains. Legal values are:

                      - "token"
                      - "pos_tagged"

    :param level1_func: The function that processes individual elements
    :param level2_func: The function that processes 2nd level lists (optional)
    :param **kwargs: Additional arguments to be passed on to the functions other
                   than the list itself
    :return: returns the output of the functions specified, applied to the
             elements of the list x.
    """
    # ==========================================================================
    levels = get_level(x, type=list_type, max_level=3)  # Get the depth of list
    try:
        # -------------------------------------------------- Level 1
        if (levels == 1):
            return level1_func(x, **kwargs)

        # -------------------------------------------------- Level 2
        elif (levels == 2):
            if level2_func is not None:
                return level2_func(x, **kwargs)
            else:
                return [level1_func(sentence, **kwargs) for sentence in x]

        # -------------------------------------------------- Level 3
        elif (levels == 3):
            processed = []
            for paragraph in x:
                processed_sentences = _multilevel_call(paragraph,
                                                       list_type,
                                                       level1_func,
                                                       level2_func,
                                                       **kwargs)
                processed.append(processed_sentences)
            return processed

    except Exception as e:
        # TODO: throw a real error message.
        print("Ohh oh! multilevel call went funny!")
        print(str(e))
        return(None)



