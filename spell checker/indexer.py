#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
from Classifaer_Iterations import Classifaer,Iteration
from ErrorModels import ErrorModel
from LanguageModel import LanguageModel
from Trie import Candidate,PrioritizedItem,Trie
from funks import flatten_dictionary,_get_levenshtein_matrix,distance
import codecs
import pandas as pd
from re import escape
import functools
import re
import os
from string import punctuation
from collections import defaultdict
import gzip

ru = list("йцукенгшщзхъфывапролджэячсмитьбюё")
en = list("qwertyuiop[]asdfghjkl;'zxcvbnm,.`")

queries_file = codecs.open('queries_all.txt',encoding = 'utf-8')
original_queries = []
fixed_queries = []
for q in queries_file:
    q=q.lower()
    if '\t' in q:
        original_queries.append(q[:(q.index('\t'))])
        fixed_queries.append(q[(q.index('\t')+1):])

punctuation = escape(punctuation)

fixed_queries_to_words = pd.Series(fixed_queries).replace('[' + punctuation + ']', '', regex=True).str.split()
fixed_words = [item for sublist in fixed_queries_to_words for item in sublist]



original_queries_to_words = pd.Series(original_queries).replace('[' + punctuation + ']', '', regex=True).str.split()
original_words = [item for sublist in original_queries_to_words for item in sublist]
error_model = ErrorModel()

for original, fixed in zip(original_queries_to_words, fixed_queries_to_words):
    number_of_words = min(len(original), len(fixed))
    for i in range(number_of_words):
        error_model.update_statistics(original[i], fixed[i])

error_model.calc_weights()

with open('lenta.pkl', 'rb') as f:
    new_wcnt = pickle.load(f)

with codecs.open('queries_all.txt',encoding = 'utf-8') as tmp:
    word_count = defaultdict(int)
    for line in tmp:
        if '\t' in line:
            line = line[(line.index('\t')+1):]

        line = re.findall(r'\w+',line)
        for i in range(len(line)):
            word_count[line[i].lower()] += 1
    for word in new_wcnt:
        word_count[word] += 1
with codecs.open('queries_all.txt',encoding = 'utf-8') as tmp:
    char_count = defaultdict(int)
    for line in tmp:
        if '\t' in line:
            line = line[(line.index('\t')+1):]
        line = ' '.join(re.findall(r'\w+',line))
        for i in range(len(line)):
            char_count[line[i].lower()] += 1
with codecs.open('queries_all.txt',encoding = 'utf-8') as tmp:
    dd_int = functools.partial(defaultdict, int)
    stat_word = defaultdict(dd_int)
    for line in tmp:
        if '\t' in line:
            line = line[(line.index('\t')+1):]
        line = re.findall(r'\w+',line)
        for i in range(len(line)-1):
            stat_word[line[i].lower()][line[i+1].lower()] += 1
with codecs.open('queries_all.txt',encoding = 'utf-8') as tmp:
    dd_int = functools.partial(defaultdict, int)
    stat_char = defaultdict(dd_int)
    for line in tmp:
        if '\t' in line:
            line = line[(line.index('\t')+1):]
        line = list(' '.join(re.findall(r'\w+',line)))
        for i in range(len(line)-1):
            stat_char[line[i].lower()][line[i+1].lower()] += 1
            
dict_for_char_proba = defaultdict(tuple)
count = 0
for i in char_count.keys():
    count += char_count[i]
for i in char_count.keys():
    dict_for_char_proba[i] = char_count[i]/count

for i in range(len(new_wcnt) -1):
    stat_word[new_wcnt[i]][new_wcnt[i+1]] += 1

dict_for_bi_word_proba = defaultdict(tuple)
for i in stat_word.keys():
    count = 0
    for j in stat_word[i].keys():
        count += stat_word[i][j]
    dict_for_bi_word_proba[i] = (count,stat_word[i])

dict_for_bi_proba = defaultdict(tuple)
for i in stat_char.keys():
    count = 0
    for j in stat_char[i].keys():
        count += stat_char[i][j]
    dict_for_bi_proba[i] = (count,stat_char[i])

w_cnt = 0
for i in word_count.values():
    w_cnt+=i

language_model= LanguageModel(w_cnt)

for word in word_count.keys():
    for i in range(word_count[word]):
        language_model.update_statistics(word)

language_model.calculate_weights()
with open('word_count.pkl', 'wb') as f:
    pickle.dump(word_count, f)
with open('dict_for_bi_word_proba.pkl', 'wb') as f:
    pickle.dump(dict_for_bi_word_proba, f)
with open('dict_for_char_proba.pkl', 'wb') as f:
    pickle.dump(dict_for_char_proba, f)
with open('error_model.pkl', 'wb') as f:
    pickle.dump(error_model, f)
with open('language_model.pkl', 'wb') as f:
    pickle.dump(language_model, f)

