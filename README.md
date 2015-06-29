Intro
------
A package with wrapper functions for the nltk package. The aim is to make the
functionality of the already great nltk package even more convenient.

With natural language processing, you will often be making use of nested lists.
So rather than having to make use of lots of nested for loops in your own
functions, these nested lists are automatically taken into account. The depth of
the nesting is automatically calculated and the appropriate nltk functions are
called on the relevant items.

Installing
----------
This package is still in the early stages of development. But if you wish to try 
it out, then run the following code into a terminal window to install it in 
development mode: 

```
pip install -e git+https://github.com/ronrest/simple_nlp.git#egg=simple_nlp
```

(You may need to add `sudo` to the begining if you are running from linux)

Using
------

### Tokenizing
```python
from simple_nlp import *
s = """ """
```

Tokenize words from the entire text, so you get a list of word strings.

```python
tokens = tokenize(s, levels_out=1)
```

Tokenize words, but also group the words by sentences. You end up with a
list that contains lists. Each of those inner lists represent the contents of
one sentence. Those inner lists contain string elements of the words.

```python
tokens = tokenize(s, levels_out=2)
```

Tokenize words, and group by sentences, but also group by paragraphs. The parent
list contains lists representing the content from each paragraph. Those lists
contain lists of sentences, which have the structure described above.

```python
tokens = tokenize(s, levels_out=3)
```

---


### POS Tagging
To perform Part Of Speech tagging, just take one of the tokenized lists we 
created above, and do:
 
```python
tagged = pos_tag(tokens)
```
 
Note, that it doesn't matter what level of nesting you chose when creating the 
tokens. `pos_tag` automatically detects the nesting used. The only thing you 
need to make sure is that all the inner most elements are a consistent depth. 
So something like `[["a","b"], [["c", "d"]]]` would not work, but 
`[[["a","b"]], [["c", "d"]]]` and `[["a","b"], ["c", "d"]]` would work. 

---

### Chunking and Named Entity Recognition
Once you have got POS tagged items, you can perform chunking on them. You can 
specify the regular expression you want to use to perform chunking with, or you 
can use a preset. By default it uses a preset that chunks for Noun Phrases. 
 
```python
chunked = chunk(tagged)
```

You can specify some other pattern by using: 

```python
chunked = chunk(tagged, pattern=myPattern)
```

Simply replace myPattern with an nltk chunking regular expression, or use one 
of the preset variable names below. 

__Presets__: 

- __CHUNK_PATTERN_NP1__
    - This pattern was taken from https://github.com/lukewrites/NP_chunking_with_nltk
    - Written by: lukewrites
    
    ```python
    CHUNKED_NP: {<JJ>*<NN>+}
                {<JJ>*<NN><CC>*<NN>+}
    ```
- __CHUNK_PATTERN_NP2__ (Default Preset)
    - This pattern was taken from https://github.com/lukewrites/NP_chunking_with_nltk
    - Written by: lukewrites
    
    ```python 
    CHUNKED_NP: {<DT><WP><VBP>*<RB>*<VBN><IN><NN>}
                {<NN|NNS|NNP|NNPS><IN>*<NN|NNS|NNP|NNPS>+}
                {<JJ>*<NN|NNS|NNP|NNPS><CC>*<NN|NNS|NNP|NNPS>+}
                {<JJ>*<NN|NNS|NNP|NNPS>+}
    ```

- __CHUNK_PATTERN_NP3__
    - This pattern was taken from http://pythonprogramming.net/chunking-nltk-tutorial/
    - Written by: Harrison Kinsley
    
    ```python
    CHUNKED_NP: {<RB.?>*<VB.?>*<NNP>+<NN>?}
    ```


You can also make use of nltk's built in Named Entity recognition, use: 

```python
ne = chunk(tagged, ne=True)
```

or 

```python
ne = named_entities(tagged)
```

By default, when it detects named entities, it labels them by the type of entity, 
eg PERSON, ORGANIZATION, FACILITY, etc. If you want to classify them based on 
whether or not they are named entities without the categories, then use: 

```python
chunk(tagged, ne=True, binary=True)
```

or 

```python
ne = named_entities(tagged, binary=True)
```



