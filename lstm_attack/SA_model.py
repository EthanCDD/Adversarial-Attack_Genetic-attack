# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 15:16:38 2020

@author: 13758
"""
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SentimentAnalysis(nn.Module):
  def __init__(self, batch_size, embedding_matrix, hidden_size, kept_prob, num_layers, bidirection, embedding_dim = 300):
    super(SentimentAnalysis, self).__init__()
    self.batch_size = batch_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.embed = nn.Embedding.from_pretrained(embedding_matrix) 
    # self.is_train = is_train
    self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size = hidden_size, num_layers = num_layers, bidirectional=bidirection)
    
    self.fc = nn.Linear(hidden_size if bidirection==False else hidden_size*2, 2)
    self.softmax = nn.Softmax(1)
    self.dropout = nn.Dropout(1-kept_prob)

  def forward(self, seqs, l, is_train):
    h_0, c_0 = self.h_c_initialisation(seqs.shape[0])
    h_0, c_0 = h_0.cuda(), c_0.cuda()
    seqs = seqs.permute(1,0)

    seqs = seqs.cuda()
    padded_seqs = self.embed(seqs).type(torch.float)
    if is_train:
      padded_seqs = self.dropout(padded_seqs)
    packed_seqs = pack_padded_sequence(padded_seqs, l)
    output, (h, c) = self.lstm(packed_seqs, (h_0, c_0))
    padded_output = pad_packed_sequence(output)
    lstm_output = torch.sum(padded_output[0], dim = 0)/l.type(torch.float).unsqueeze(1)
    # output, (h, c) = self.lstm(padded_seqs)
    # lstm_output = torch.mean(output, dim = 0)
    if is_train:
      lstm_output = self.dropout(lstm_output)
    output = self.fc(lstm_output)
    pred_out = self.softmax(output)
    return output, pred_out
    
  def h_c_initialisation(self, batch):
    h = torch.zeros(self.num_layers, batch, self.hidden_size)
    c = torch.zeros(self.num_layers, batch, self.hidden_size)

    h = h.type(torch.float)
    c = c.type(torch.float)
    return h, c

  def evaluate_accuracy(self, pred, target):
    # v = np.sum(np.argmax(pred, 1) == np.argmax(target, 1))
    v = np.sum(np.argmax(pred, 1) == target)
    return v/len(target)

  def pred(self, test_seq, l, is_train):
    pred = self.forward(test_seq, l, is_train)
    return pred