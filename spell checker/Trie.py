#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from dataclasses import dataclass, field
from typing import Any
from heapq import heappush,heappop
from funks import flatten_dictionary,_get_levenshtein_matrix,distance

class Candidate:

    def __init__(self, word, error_weight, language_weight):
        self.word = word
        self.error_weight = error_weight
        self.language_weight = language_weight

    def __repr__(self):
        return str(self.__dict__)

@dataclass(order=True)
class PrioritizedItem:
    weight : float
    result: Any=field(compare=False)
    prefix: Any=field(compare=False)
    node: Any=field(compare=False)

        
        
class Trie():
    def __init__(self,error_model, language_model,word_count):
        self.child = {}
        self.best_insert = sorted(error_model.weights[''].items(),key = lambda x:x[1])[:40]
        self.candidates = []
        self.error_model = error_model
        self.word_count = word_count
        self.language_model = language_model
        self.ru = list("йцукенгшщзхъфывапролджэячсмитьбюё")
        self.en = list("qwertyuiop[]asdfghjkl;'zxcvbnm,.`")
    def insert(self, word):
        current = self.child
        for l in word:
            if l not in current:
                current[l] = {}
            current = current[l]
        current['word'] = word
        current['lang_weight'] = self.language_model.weights[word]
    def search(self, word):
        current = self.child
        for l in word:
            if l not in current:
                return False
            current = current[l]
        if 'word' not in current:
            return False
        return current['word']
    def fit(self):
        words = set(self.word_count.keys())
        for word in words:
            self.insert(word)
            
    def modify_candidates_list(self, candidate):
        new_candidate = True
        for cand in self.candidates:
            if cand.word == candidate.word:
                new_candidate = False
                cand.language_weight = max(candidate.language_weight, cand.language_weight)
                cand.error_weight = min(candidate.error_weight, cand.error_weight)
        if new_candidate and len(self.candidates) <= self.max_number_of_candidates:
            self.candidates.append(candidate)
    def last_insert(self,candidates):
        cond = candidates[:]
        for val in candidates:
            for k in self.best_insert:
                cond.append(Candidate(val.word + k[0],val.error_weight + k[1],
                                           self.language_model.weights[val.word + k[0]]))
        return cond        
    def can_be_added(self, weight):
        return (weight < self.max_sum_of_weights and len(self.candidates) <= self.max_number_of_candidates)
    
    def find_candidates(self, prefix, result='', root=None, weight=0, part=1):  # returns number of occurencies

        if root is None:
            root = self.child
        node = root

        if len(prefix) < 1:
            if 'word' in node:
                self.modify_candidates_list(Candidate(result, weight, node['lang_weight']))
            return
        

        c = prefix[0]
        
        
        for key in node:
            if (key not in ['word','lang_weight']) and (not part or key in ru+en):
                if len(prefix) >= 2:
                    if (prefix[1] == key):
                        if (prefix[0] in node[key]):
                            additional_weight = 1.5
                            if self.can_be_added(weight + additional_weight):
                                    self.find_candidates(prefix[2:], result + prefix[1::-1], node[key][prefix[0]],
                                                         weight + additional_weight)
                if (c in self.ru) and (key == self.en[self.ru.index(c)]):
                    additional_weight = 0.1
                    if self.can_be_added(weight + additional_weight):
                        self.find_candidates(prefix[1:], result + self.en[self.ru.index(c)], node[key], weight + additional_weight)

                if (c in self.en) and (key == self.ru[self.en.index(c)]):
                    additional_weight = 0.1
                    if self.can_be_added(weight + additional_weight):
                        self.find_candidates(prefix[1:], result + self.ru[self.en.index(c)], node[key], weight + additional_weight)


                if key == c:
                    self.find_candidates(prefix[1:], result + c, node[key], weight)
                    # insert
                    additional_weight = self.error_model.weights[''][c]*0.3
                    if self.can_be_added(weight + additional_weight):
                        self.find_candidates(prefix, result + c, node[key], weight + additional_weight)

                else:

                    if key in self.error_model.weights['']:
                        additional_weight = self.error_model.weights[''][key]*0.3
                        if self.can_be_added(weight + additional_weight):
                            self.find_candidates(
                                prefix, result + key, node[key], weight + additional_weight)

                    if key in self.error_model.weights[c]:
                        additional_weight = self.error_model.weights[c][key]
                        if self.can_be_added(weight + additional_weight):
                            self.find_candidates(
                                prefix[1:], result + key, node[key], weight + additional_weight)
        if '' in self.error_model.weights[c]:
            additional_weight = self.error_model.weights[c]['']
            if self.can_be_added(weight + additional_weight):
                self.find_candidates(prefix[1:], result, node, weight + additional_weight)
            return
    
    def flow(self,prefix):
        self.queue = []
        current = self.child
        heappush(self.queue, PrioritizedItem(0,'',prefix,current))
        vr = 0
        while(len(self.queue)>0):
            vr += 1
            val = heappop(self.queue)
            if len(val.prefix) < 1:
                
                if 'word' in val.node:
                    self.modify_candidates_list(Candidate(val.result, val.weight, val.node['lang_weight']))
                continue
        
            c = val.prefix[0]
            for key in val.node:
                if key not in ['word','lang_weight']:
                    if len(val.prefix) >= 2:
                        if (val.prefix[1] == key):
                            if (val.prefix[0] in val.node[key]):
                                additional_weight = 1.5
                                if self.can_be_added(val.weight + additional_weight):
                                       heappush(self.queue,PrioritizedItem(val.weight + additional_weight,
                                                                           val.result + prefix[1::-1],
                                                                           val.prefix[2:],
                                                                           val.node[key][val.prefix[0]]))
                    if (c in self.ru) and (key == self.en[self.ru.index(c)]):
                        additional_weight = 0.1
                        if self.can_be_added(val.weight + additional_weight):
                            heappush(self.queue,PrioritizedItem(val.weight + additional_weight,
                                                                               val.result + self.en[self.ru.index(c)],
                                                                               val.prefix[1:], 
                                                                               val.node[key]))

                    if (c in self.en) and (key == self.ru[self.en.index(c)]):
                        additional_weight = 0.1
                        if self.can_be_added(val.weight + additional_weight):
                            heappush(self.queue,PrioritizedItem(val.weight + additional_weight,
                                                                               val.result + self.ru[self.en.index(c)],
                                                                               val.prefix[1:], 
                                                                               val.node[key]))
                    if key == c:
                        heappush(self.queue,PrioritizedItem(val.weight,
                                                            val.result + c,
                                                            val.prefix[1:], 
                                                            val.node[key]))
                    else:

                        if key in self.error_model.weights['']:
                            additional_weight = self.error_model.weights[''][key]*0.3
                            if self.can_be_added(val.weight + additional_weight):
                                heappush(self.queue,PrioritizedItem(val.weight + additional_weight,
                                                                val.result + key,
                                                                val.prefix, 
                                                                val.node[key]))
                                
                        if key in self.error_model.weights[c]:
                            additional_weight = self.error_model.weights[c][key]
                            if self.can_be_added(val.weight + additional_weight):
                                heappush(self.queue,PrioritizedItem(val.weight + additional_weight,
                                                                val.result + key,
                                                                val.prefix[1:], 
                                                                val.node[key]))
    
            if '' in self.error_model.weights[c]:
                additional_weight = self.error_model.weights[c]['']
                if self.can_be_added(val.weight + additional_weight):
                    heappush(self.queue,PrioritizedItem(val.weight + additional_weight,
                                                            val.result,
                                                            val.prefix[1:], 
                                                            val.node))
            continue
                             
                   
                
                
        
        

                         
    def generate(self, word, max_number_of_candidates=5, max_sum_of_weights=10, part=True):
        self.max_number_of_candidates = max_number_of_candidates
        self.max_sum_of_weights = max_sum_of_weights
        self.candidates = []
        self.flow(word)
        return self.candidates

