# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 15:54:12 2020

@author: 13758
"""
import os
import re
import numpy as np
from collections import OrderedDict, defaultdict


class pre_processing(object):
  def __init__(self, seqs):
    self.test_seqs = seqs

  def bert_tokenize(self, tokenizer):

    self.test_tokens = [tokenizer.tokenize(seq) for seq in self.test_seqs]

  def numerical(self, tokenizer, test_seqs):

    test_text = [[tokenizer.cls_token_id]+ seq + [tokenizer.sep_token_id] for seq in test_seqs]
    return test_text
  
  def bert_indx(self, tokenizer):
    test_seqs = [tokenizer.convert_tokens_to_ids(seq) for seq in self.test_tokens]
    return test_seqs