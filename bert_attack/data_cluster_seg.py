# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 15:23:49 2020

@author: 13758
"""
import numpy as np
from torch.utils.data import Dataset

class Data_infor(Dataset):
  def __init__(self, seqs, target):
    self.seqs = seqs
    self.target = target

  def __getitem__(self, indx):
    seq = self.seqs[indx]
    l = np.sum(np.sign(self.seqs[indx]))
    label = self.target[indx]
    # if label == 1:
    #   label = np.array([0, 1])
    # else:
    #   label = np.array([1, 0])
    return seq, l, label

  def __len__(self):
    return len(self.seqs)
