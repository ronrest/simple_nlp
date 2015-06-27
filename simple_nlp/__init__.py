"""====================================================
                    DESCRIPTION

=======================================================
"""
__author__ = 'ronny'

import nltk

# ==============================================================================
#                                                                       TOKENIZE
# ==============================================================================
def tokenize(text, levels_out=1):
    """
    Takes a string of text, and returns a list of tokenized words.

    You can chose to have the tokenized words nested at different levels if you
    wish to group by sentences (levels_out=2), or paragraphs (levels_out=3)
    preserve

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
    else:
        # TODO: throw some kind of exception about incorrect value.
        print("levels_out must be 1,2 or 3.")
        return(None)
