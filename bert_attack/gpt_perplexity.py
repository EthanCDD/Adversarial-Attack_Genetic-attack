# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 20:40:41 2020

@author: 13758
"""
import math
import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class gpt_2_get_words_probs(object):

  def __init__(self):
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.model = GPT2LMHeadModel.from_pretrained('gpt2')
    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  def get_probs(self, word_list):
    prefix_seq = [self.seq(seq) for seq in word_list]

    loss = [math.exp(self.score(seq)) for seq in prefix_seq]

    return loss

  def score(self, prefix_seq):
    indexed_tokens = self.tokenizer.encode(prefix_seq)
    tokens_tensor = torch.tensor([indexed_tokens])
    self.model.eval()
    
    tokens_tensor = tokens_tensor.to(self.device)
    self.model.to(self.device)

    # Predict all tokens
    with torch.no_grad():
      outputs = self.model(input_ids=tokens_tensor, labels = tokens_tensor)[0]
    return outputs
    
  def seq(self, prefix):
    sentence = ''

    for word in prefix:
        sentence += word + ' '
    sentence = sentence.strip()
    return sentence
  