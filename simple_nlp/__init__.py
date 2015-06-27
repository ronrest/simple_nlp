"""====================================================
                    DESCRIPTION

=======================================================
"""
__author__ = 'ronny'

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
def pos_tag(tokens):
    """
    Takes a list of tokens and returns a list of Part of Speech (POS) Tagged
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
    # TODO: make it accept a single string that is not nested in a list.
    assert isinstance(tokens, list), \
        "Argument *tokens* in pos_tag() function must be a list."
    try:
        levels = get_level(tokens)
    except:
        #TODO: throw a propper error message.
        print("Something is wrong with the tokens list")
    assert isinstance(levels, int), \
        "Something wrong with the *tokens* list argument in pos_tag() function"
    assert (levels>=1) and (levels<=3), \
        "The depth of levels for *tokens* list must be in the range [1, 3]"

    if (levels == 1):
        return(nltk.pos_tag(tokens))
    elif (levels == 2):
        return(nltk.pos_tag_sents(tokens))
    elif (levels == 3):
        tagged = []
        for paragraph in tokens:
            tagged_sentences = nltk.pos_tag_sents(paragraph)
            tagged.append(tagged_sentences)
        return (tagged)
    else:
        #TODO: throw some error messsage
        print("Something went wrong with pos_tag(). Double check your arguments.")
        return(None)








# ==============================================================================
#                                                                      GET LEVEL
# ==============================================================================
def get_level(x):
    """
    Takes a list of token strings, and returns how many levels deep the tokens
    are.

    :param x: (list) The list of tokens
    :return: (int) an integer representing how many levels deep the string
              tokens are.
    """
    assert isinstance(x, list), "Argument *x* in get_level must be a list"

    #TODO: test that all elements of the list are consistently the same depth.

    current_nest = x[0]
    level = 1
    while level <= 3:
        if isinstance(current_nest, str):
            break
        elif not isinstance(current_nest, list):
            #TODO: throw a type exception
            print("x must be a nested list of strings")
            return(None)
        else:
            current_nest = current_nest[0]
            level += 1
    if (level > 3):
        #TODO: throw an error exception
        print("You are in too deep! get_level() only processes up to 3 levels deep.")
        return(None)
    return(level)
