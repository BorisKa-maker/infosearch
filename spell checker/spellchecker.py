#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pickle
from Classifaer_Iterations import Classifaer,Iteration
from ErrorModels import ErrorModel
from LanguageModel import LanguageModel
from Trie import Candidate,PrioritizedItem,Trie


if __name__ == '__main__':
    with open('word_count.pkl', 'rb') as f:
        word_count = pickle.load(f)
    with open('dict_for_bi_word_proba.pkl', 'rb') as f:
        dict_for_bi_word_proba = pickle.load(f)
    with open('dict_for_char_proba.pkl', 'rb') as f:
        dict_for_char_proba = pickle.load(f)
    with open('error_model.pkl', 'rb') as f:
        error_model = pickle.load(f)
    with open('language_model.pkl', 'rb') as f:
        language_model = pickle.load(f)

        
trie = Trie(error_model, language_model,word_count)
trie.fit()
spellchecker = Iteration(word_count,dict_for_bi_word_proba,dict_for_char_proba,trie,language_model)
while (True):
    try:
        query =input()
    except (EOFError):
        break
    print(spellchecker.iterations(query))

