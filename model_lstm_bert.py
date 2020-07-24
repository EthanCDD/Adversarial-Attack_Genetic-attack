# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 16:46:24 2020

@author: 13758
"""
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class bert_lstm(nn.Module):
  def __init__(self, bert, output_size, bidirection, n_layer, hidden_size, freeze=True, kept_prob = 1):
    super().__init__()
    self.n_layer = n_layer
    self.hidden_size = hidden_size
    self.bert = bert
    if freeze:
        self.freeze_bert()
    embedding_size = bert.config.to_dict()['hidden_size']
    self.lstm = nn.LSTM( input_size=embedding_size,
                hidden_size=hidden_size, 
                num_layers = n_layer, 
#                dropout = 1-kept_prob if n_layer>2 else 0,
                bidirectional = bidirection)
    if bidirection == True:
        bidirect = 2
    else:
        bidirect = 1    
    self.fc = nn.Linear(hidden_size*bidirect, output_size)
    self.dropout = nn.Dropout(1-kept_prob)
    self.softmax = nn.Softmax(1)
    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  def forward(self, text, l, is_train):
    h_0, c_0 = self.h_c_initialisation(text.shape[0])
    h_0, c_0 = h_0.cuda(), c_0.cuda()

    embedding_mat = self.bert(text)[0]
    batch_size, seq_len, emb_dim = embedding_mat.shape
    
#    h_0, c_0 = self.hidden_b(batch_size)
    bert_seqs = embedding_mat.permute(1,0,2)
    if is_train:
        bert_seqs = self.dropout(bert_seqs)
    packed_seqs = pack_padded_sequence(bert_seqs, l)
    output, (h,c) = self.lstm(packed_seqs, (h_0, c_0))
    output = pad_packed_sequence(output)
    
    lstm_output = torch.sum(output[0], dim = 0)/l.type(torch.float).unsqueeze(1)
    if is_train:
        lstm_output = self.dropout(lstm_output)
    output = self.fc(lstm_output)
    pred_out = self.softmax(output)

    return output, pred_out

  def h_c_initialisation(self, batch):
    h = torch.zeros(self.n_layer, batch, self.hidden_size)
    c = torch.zeros(self.n_layer, batch, self.hidden_size)

    h = h.type(torch.float)
    c = c.type(torch.float)
    return h, c


  def freeze_bert(self):
    for cont in self.bert.parameters():
      cont.requires_grad = False
    
  def evaluate_accuracy(self, pred, target):
    v = np.sum(np.argmax(pred, 1) == target)
    return v/len(target)

  def pred(self, seq, l, is_train):
    return self.forward(seq, l, is_train)
