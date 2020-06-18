# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 15:21:06 2020

@author: 13758
"""

import numpy as np
import os
import gc
import pickle
import data_cluster_seg
from torch.nn.utils.data import Subset, DataLoader
import torch.nn as nn
import torch
import SA_model
from keras.preprocessing.sequence import pad_sequences


rnn_state_save = os.path.join('/content/drive/My Drive/Master_Final_Project/Genetic_attack/Code/nlp_adversarial_example_master_pytorch','best_SA')

MAX_VOCAB_SIZE = 50000
with open(('aux_files/dataset_%d.pkl' %MAX_VOCAB_SIZE), 'rb') as f:
    dataset = pickle.load(f)

max_len = 250
padded_train_raw = pad_sequences(dataset.train_seqs2, maxlen = max_len, padding = 'post')
padded_test_raw = pad_sequences(dataset.test_seqs2, maxlen = max_len, padding = 'post')

# TrainSet
data_set = data_cluster_seg.Data_infor(padded_train_raw, dataset.train_y)
num_train = len(data_set)
indx = list(range(num_train))
train_set = Subset(data_set, indx)

# TestSet
data_set = data_cluster_seg.Data_infor(padded_test_raw, dataset.test_y)
num_test = len(data_set)
indx = list(range(num_test))
test_set = Subset(data_set, indx)

# [300, 50001] 'UKN':50000; The first has no meaning
embedding_matrix = np.load('aux_files/embeddings_glove_%d.npy' %(MAX_VOCAB_SIZE))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 64
hidden_size = 128
train_loader = DataLoader(train_set, batch_size = batch_size, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_set, batch_size = int(num_test/3), shuffle = True, pin_memory=True)


embedding_matrix = torch.tensor(embedding_matrix.T).to(device)
rnn = SA_model.SentimentAnalysis(batch_size, embedding_matrix, hidden_size, 0.7)
rnn = rnn.to(device)

learning_rate = 0.0005
criterion = nn.CrossEntropyLoss()
optimiser = torch.optim.AdamW(rnn.parameters(), lr = learning_rate)

epoches = 200
best_epoch = 0
best_acc = 0
patience = 20

for epoch in range(epoches):
  test_pred = torch.tensor([])
  test_targets = torch.tensor([])
  train_pred = torch.tensor([])
  train_targets = torch.tensor([])
  test_loss = []
  train_loss = []
  
  rnn.train()
  for batch_index, (seqs, length, target) in enumerate(train_loader):
    seqs, target, length = seqs.to(device), target.to(device), length.to(device)
    seqs = seqs.type(torch.LongTensor)
    len_order = torch.argsort(length, descending = True)
    length = length[len_order]
    seqs = seqs[len_order]
    target = target[len_order]
    optimiser.zero_grad()
    output = rnn(seqs, length)
    loss = criterion(output, target)
    loss.backward()
    optimiser.step()

    train_pred = torch.cat((train_pred, output.cpu()), dim = 0)
    train_targets = torch.cat((train_targets, target.type(torch.float).cpu()))
    train_loss.append(loss)
    
    if batch_index % 100 == 0:
      print('Train Batch:{}, Train Loss:{:.4f}.'.format(batch_index, loss.item()))

  
  train_accuracy = rnn.evaluate_accuracy(train_pred.detach().numpy(), train_targets.detach().numpy())
  print('Epoch:{}, Train Accuracy:{:.4f}, Train Mean loss:{:.4f}.'.format(epoch, train_accuracy, sum(train_loss)/len(train_loss)))


  rnn.eval()
  with torch.no_grad():
    for batch_index, (seqs, length, target) in enumerate(test_loader):
      seqs, target, length = seqs.to(device), target.to(device), length.to(device)
      seqs = seqs.type(torch.LongTensor)
      len_order = torch.argsort(length, descending = True)
      length = length[len_order]
      seqs = seqs[len_order]
      target = target[len_order]
      output = rnn(seqs, length)
      test_pred = torch.cat((test_pred, output.cpu()), dim = 0)
      test_targets = torch.cat((test_targets, target.type(torch.float).cpu()))
      loss = criterion(output, target)
      test_loss.append(loss.item())
      if batch_index % 30 == 0:
        print('Test Batch:{}, Test Loss:{:.4f}.'.format(batch_index, loss.item()))
    accuracy = rnn.evaluate_accuracy(test_pred.numpy(), test_targets.numpy())
    print('Epoch:{}, Test Accuracy:{:.4f}, Test Mean loss:{:.4f}.'.format(epoch, accuracy, sum(test_loss)/len(test_loss)))
    # best save
    if accuracy > best_acc:
      best_acc = accuracy
      best_epoch = epoch
      torch.save(rnn.state_dict(), rnn_state_save)
    # early stop
    if epoch-best_epoch >=patience:
      print('Early stopping')
      print('Best epoch: {}, Best accuracy: {:.4f}.'.format(best_epoch, best_acc))
      break
  
gc.collect()
torch.cuda.empty_cache()






