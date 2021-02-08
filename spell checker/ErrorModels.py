#!/usr/bin/env python
# coding: utf-8

# In[1]:
from collections import defaultdict
import functools
import numpy as np 
import math

class ErrorModel:

    def __init__(self):
        dd_int = functools.partial(defaultdict, int)
        self.statistics = defaultdict(dd_int)
    
    def update_statistics(self, given, fixed):
        n, m = len(given), len(fixed)
        MAX_DISTANCE = (n + 1) * (m + 1)

        matrix=_get_levenshtein_matrix(given,fixed)

        position = [n, m, matrix[m, n]] 
        while position[2] != 0: 
            x, y = position[0], position[1]

            possible_actions = [matrix[y - 1][x - 1] if (x > 0) and (y > 0) else MAX_DISTANCE,  # change
                                matrix[y - 1][x] if y > 0 else MAX_DISTANCE,  # add
                                matrix[y][x - 1] if x > 0 else MAX_DISTANCE]  # delete
            action = np.argmin(possible_actions)

            if action == 0:  # change
                if position[2] != possible_actions[action.item()]:
                    position[2] -= 1
                    self.statistics[given[x-1]][fixed[y-1]] += 1
                position[0] -= 1
                position[1] -= 1
            elif action == 1:  # add
                if position[2] != possible_actions[action.item()]:
                    position[2] -= 1
                    self.statistics[''][fixed[y - 1]] += 1
                position[1] -= 1
            else:  # delete
                if position[2] != possible_actions[action.item()]:
                    position[2] -= 1
                    self.statistics[given[x - 1]][''] += 1
                position[0] -= 1
    
    def calc_weights(self,alpha = 0.1):
        def_w = 10
        for k in self.statistics.keys():
            w = np.sum(list(self.statistics[k].values()))
            for k1 in self.statistics[k]:
                if self.statistics[k][k1]/w < def_w:
                    def_w = self.statistics[k][k1]/w
        def_w = math.log(def_w)*(-1)
        dd_d = functools.partial(defaultdict, float)
        self.weights = defaultdict(dd_d)
        for k in self.statistics.keys():
            w = np.sum(list(self.statistics[k].values()))
            for k1 in self.statistics[k]:
                self.weights[k][k1] = math.log(self.statistics[k][k1]/w)*(-1)
def _get_levenshtein_matrix(a, b):
    n, m = len(a), len(b)
    inverse = False
    if n > m:
        a, b = b, a
        n, m = m, n
        inverse = True

    current_row = list(range(n + 1))  
    matrix = np.array([current_row])
    for i in range(1, m + 1):
        previous_row, current_row = current_row, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete, change = previous_row[j] + 1,                                   current_row[j - 1] + 1,                                   previous_row[j - 1] + int(a[j - 1] != b[i - 1])
            current_row[j] = min(add, delete, change)
        matrix = np.vstack((matrix, [current_row]))
    return matrix if not inverse else matrix.T

