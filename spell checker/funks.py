#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def flatten_dictionary(dictionary):
    values = []
    for d in dictionary.values():
        for v in d.values():
            values.append(v)
    return values

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

def distance(a, b):
    "Calculates the Levenshtein distance between a and b."
    n, m = len(a), len(b)
    if n > m:
        # Make sure n <= m, to use O(min(n, m)) space
        a, b = b, a
        n, m = m, n

    current_row = range(n + 1)  # Keep current and previous row, not entire matrix
    for i in range(1, m + 1):
        previous_row, current_row = current_row, [i] + [0] * n
        for j in range(1, n + 1):
            add, delete, change = previous_row[j] + 1, current_row[j - 1] + 1, previous_row[j - 1]
            if a[j - 1] != b[i - 1]:
                change += 1
            current_row[j] = min(add, delete, change)

    return current_row[n]

