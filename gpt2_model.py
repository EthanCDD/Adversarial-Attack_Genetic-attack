# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 11:25:18 2020

@author: 13758
"""
import torch
import numpy as np
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class gpt_2_get_words_probs(object):

  def __init__(self):
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.model = GPT2LMHeadModel.from_pretrained('gpt2')
    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  def get_probs(self, prefix, nearest_w, n, loc):
    prefix = self.seq(prefix, n, loc)
    nearest_w_space = [' '+x for x in nearest_w]
    indexs_near, tokens_near = self.near_w(nearest_w_space)

    indexed_tokens = self.tokenizer.encode(prefix)
    tokens_tensor = torch.tensor([indexed_tokens])
    self.model.eval()
    
    tokens_tensor = tokens_tensor.to(self.device)
    self.model.to(self.device)

    # Predict all tokens
    with torch.no_grad():
        outputs = self.model(tokens_tensor)
        predictions = outputs[0]
    predicted_index = torch.argsort(-predictions[0, -1, :]).cpu().numpy()
    # list_order = sorted([int(np.where(predicted_index == i)[0]) for i in indexs_near])
    order = np.argsort(  np.array([int(np.where(predicted_index == i)[0]) for i in indexs_near]) )
#    order = np.argsort(  np.array([int(np.where(predicted_index == i)[0]) if i <=50257 else i for i in indexs_near]) )
#    for i in range(10):
#      print(self.tokenizer.decode(int(predicted_index[i])))
    return order
    
  def seq(self, prefix, n, loc):
    sentence = ''
    if loc > n:
      for word in prefix:
        sentence = sentence+' '+word
    else:
      for word in prefix:
          sentence += word + ' '
      sentence = (self.tokenizer.bos_token +' '+ sentence).strip()
    return sentence
  
  def near_w(self, nearest_w):
    dict_select = {}
    for item in nearest_w:
      token_indx = self.tokenizer.encode(item)

      if len(token_indx)==1:
        dict_select[item] = token_indx[0]
      else:
        dict_select[item] = self.tokenizer.unk_token_id#50258
    tokens = list(dict_select.keys())
    indexs = list(dict_select.values())

    return indexs, tokens