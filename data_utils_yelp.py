# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 18:20:54 2020

@author: 13758
"""

import os
import pickle
from keras.preprocessing.text import Tokenizer
class YELPDataset(object):
  def __init__(self, path='yelp_dataset_list', max_vocab_size=None):
    with open('yelp_dataset_list', 'rb') as f:
      data = pickle.load(f)
    train_text, self.train_y = data[0], data[1]
    test_text, self.test_y = data[2], data[3]
    self.train_text, self.test_text = self.read_text(train_text, test_text)
    self.tokenizer = Tokenizer()
    print('Tokenizing')
    self.tokenizer.fit_on_texts(self.train_text)
    if max_vocab_size is None:
        max_vocab_size = len(self.tokenizer.word_index) + 1
    #sorted_words = sorted([x for x in self.tokenizer.word_counts])
    #self.top_words = sorted_words[:max_vocab_size-1]
    #self.other_words = sorted_words[max_vocab_size-1:]
    self.dict = dict()
    self.train_seqs = self.tokenizer.texts_to_sequences(self.train_text)
    self.train_seqs2 = [[w if w < max_vocab_size else max_vocab_size for w in doc] for doc in self.train_seqs]
    
    self.test_seqs = self.tokenizer.texts_to_sequences(self.test_text)
    self.test_seqs2 = [[w if w < max_vocab_size else max_vocab_size for w in doc] for doc in self.test_seqs]
    
    self.dict['UNK'] = max_vocab_size
    self.inv_dict = dict()
    self.inv_dict[max_vocab_size] = 'UNK'
    self.full_dict = dict()
    self.inv_full_dict = dict()
    for word, idx in self.tokenizer.word_index.items():
        if idx < max_vocab_size:
            self.inv_dict[idx] = word
            self.dict[word] = idx
        self.full_dict[word] = idx
        self.inv_full_dict[idx] = word 
    print('Dataset built !')

  def read_text(self, train_data, test_data):
        """ Returns a list of text documents and a list of their labels
        (pos = +1, neg = 0) """
        train_list = [x.lower() for x in train_data]
        test_list = [x.lower() for x in test_data]
        return train_list, test_list