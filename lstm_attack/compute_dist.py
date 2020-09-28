# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 15:36:04 2020

@author: 13758
"""
import numpy as np
MAX_VOCAB_SIZE = 50000
embedding_matrix = np.load(('aux_files/embeddings_counter_%d.npy' %(MAX_VOCAB_SIZE)))
a = np.sum(np.square(embedding_matrix), axis = 0).reshape((1, -1))
b = a.T
def compute_dis(word):
    c_ = -2*np.dot(np.expand_dims(embedding_matrix[:, word], axis = 0), embedding_matrix)
    dist = a+b[word]+c_
    return dist.squeeze()