#!/usr/bin/env python
# coding: utf-8

# In[1]:


import itertools
import numpy as np
from funks import flatten_dictionary,_get_levenshtein_matrix,distance
import math
from Trie import Candidate
class Classifaer():
    def __init__(self,word_count,dict_for_bi_word_proba,dict_for_char_proba,w_cnt):
        self.word_count = word_count
        self.dict_for_bi_word_proba = dict_for_bi_word_proba
        self.dict_for_char_proba = dict_for_char_proba
        self.w_cnt = w_cnt
        
    
    def naive_word_classifaer(self,request,alpha = 10E-20,betta = 0.1):
        words = request.split()

        list_ = (np.asarray([self.word_count[w] for w in words]) + alpha)/(self.w_cnt + alpha + betta)
        return (request, np.log(list_).sum())
    
    def word_classifier(self,request,alpha = 10E-20,betta = 60):
        request = request.split()
        if request[0] in self.word_count:
            proba = math.log((self.word_count[request[0]]+alpha)/(self.w_cnt+alpha+betta))
        else:
            proba = math.log((alpha)/(self.w_cnt + alpha+betta))
        for i in range(len(request)-1):
            if self.dict_for_bi_word_proba[request[i]] or None is not None:
                count = self.dict_for_bi_word_proba[request[i]][0]
                if request[i+1] in self.dict_for_bi_word_proba[request[i]][1]:
                    val = self.dict_for_bi_word_proba[request[i]][1][request[i+1]]
                else:
                    val = 0
                proba += math.log((val + alpha)/(count + alpha + betta))
            else:
                proba += math.log((alpha)/(alpha+betta))
        return (' '.join(request),proba)  
    
    def char_classifier(self,request,alpha = 10e-20,betta = 0.1):
        proba = math.log(self.dict_for_char_proba[request[0]])
        for i in range(len(request)-1):
            count,val = self.dict_for_bi_proba[request[i]][0],self.dict_for_bi_proba[request[i]][1][request[i+1]]
            proba += math.log((val + alpha)/(count + alpha + betta))
        return (request,proba)
    def word_classifier_n(self,request,pred,alpha = 10E-20,betta = 0.1):
        proba = 0
        for i in range(len(request)-1):
            if self.dict_for_bi_word_proba[request[i].word] or None is not None:
                count = self.dict_for_bi_word_proba[request[i].word][0]
                if request[i+1].word in self.dict_for_bi_word_proba[request[i].word][1]:
                    val = self.dict_for_bi_word_proba[request[i].word][1][request[i+1].word]
                else:
                    val = 0
                proba += math.log((val + alpha)/(count + alpha + betta))
            else:
                proba += math.log((alpha)/(alpha+betta))
        key= lambda x: 1.15*x.language_weight+x.error_weight + distance(x.word,word)*2.5/len(word)
        post = ' '.join([x.word for x in request])
        pred = ' '.join(pred)
        return (post,proba -  distance(pred,post)/len(pred))  


class Iteration():
    def __init__(self,word_count,dict_for_bi_word_proba,dict_for_char_proba,trie,language_model):
        self.clsf = Classifaer(word_count,dict_for_bi_word_proba,dict_for_char_proba,language_model.w_cnt)
        self.trie = trie
        self.language_model = language_model
    
    def split_generator(self,request):
        indexes = [i for i in range(len(request)) if request[i] == ' ']
        indexes = [indexes[i] - i - 1 for i in range(len(indexes))]
        request = list(''.join(request.split()))
        l = [i for i in range(0,len(request)-1) if i not in indexes]
        all_comb = [itertools.combinations(l,i) for i in range(2)]
        strings = []
        for combins in all_comb:
            for comb in combins:
                new_req = list(request)
                comb = sorted(list(comb)+indexes)
                k=0
                for i in comb:
                    new_req.insert(i+k+1,' ')
                    k+=1
                strings.append(''.join(new_req))
        return strings
    def join_generator(self,request):
        indexes = [i for i in range(len(request)) if request[i] == ' ']
        indexes = [indexes[i] - i -1 for i in range(len(indexes))]
        request = list(''.join(request.split()))
        x = len(indexes)-1
        all_comb = [itertools.combinations(indexes,i) for i in range(x if x > 0 else 0,len(indexes)+1)]
        strings = []
        for combins in all_comb:
            for comb in combins:
                new_req = list(request)
                k=0
                for i in comb:
                    new_req.insert(i+k+1,' ')
                    k+=1
                strings.append(''.join(new_req))
        return strings
    
    def j_c(self,request):
        kr = self.join_generator(request)
        res = []
        for i in kr:
            res.append(self.clsf.naive_word_classifaer(i))
        return max(res,key = lambda x:x[1])
    
    def s_c(self,request):
        kr = self.split_generator(request)
        res = []
        for i in kr:
            res.append(self.clsf.naive_word_classifaer(i))
        return max(res,key = lambda x:x[1])
    
    def orpho_generator(self,request):
        request = request.split()
        strings = []
        fixed_words = []
        for word in request:
            if len(word)<=10:
                candidates = self.trie.generate(word, max_number_of_candidates=60, max_sum_of_weights=0.4*len(word))
                if len(candidates) < 1:
                    candidates = [Candidate(word=word, language_weight=30, error_weight=0)]
                cand = self.trie.last_insert(sorted(candidates, key= lambda x: 1.15* x.language_weight                                                   + x.error_weight + distance(word,x.word)*2.5/len(word))[:10])
                l1 = [[x,1.15*x.language_weight+x.error_weight +                                          distance(x.word,word)*2.5/len(word)] for x in cand]
                tmp = (sorted(l1,key = lambda x: x[1]))[:3]
                fixed_words.append([x[0] for x in tmp])
            else:
                fixed_words.append([Candidate(word=word, language_weight=self.language_model.weights[word], error_weight=0)]*3)
        x = [0,1,2]
        combins = [p for p in itertools.product(x, repeat=len(request))]
        res = []
        for comb in combins:
            res.append(self.clsf.word_classifier_n([x[comb[i]] for i,x in enumerate(fixed_words)],request))
        return max(res, key = lambda x: x[1])[0]
    
    def iterations(self,requests):
        requests = set([requests])
        i = 0
        old_request = {}
        while(i==0 or len(requests)>1):
            if i == 4:
                break
            new_requests = set()
            for request in requests:
                r1 = self.s_c(request)[0]
                if r1 not in old_request.keys():
                    old_request[r1] = self.clsf.naive_word_classifaer(r1)[1]
                    new_requests.add(r1)
                r2 = self.j_c(request)[0]
                if r2 not in old_request.keys():
                    old_request[r2] = self.clsf.naive_word_classifaer(r2)[1]
                    new_requests.add(r2)
                r3 = self.orpho_generator(request)
                if r3 not in old_request.keys():
                    old_request[r3] = self.clsf.naive_word_classifaer(r3)[1]
                    new_requests.add(r3)
            requests = new_requests
            i += 1
        return max(old_request.items(),key = lambda x:x[1])[0]
    

