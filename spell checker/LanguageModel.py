#!/usr/bin/env python
# coding: utf-8

# In[4]:


from collections import defaultdict
import math
class LanguageModel:

    def __init__(self,wcnt):
        self.statistics = defaultdict(int)
        self.size=0
        self.w_cnt = wcnt
        
    def update_statistics(self, token):
        self.statistics[token]+=1
        self.size+=1
    def muala(self):
        return (-1)*math.log(1/self.size)
    
    def calculate_weights(self):
        self.weights=defaultdict(self.muala)
        for item,stat in self.statistics.items():
            self.weights[item]=(-1)*math.log(stat/self.size)

