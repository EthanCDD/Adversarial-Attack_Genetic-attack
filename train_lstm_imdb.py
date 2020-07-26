# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 19:11:45 2020

@author: 13758
"""
import os
import torch
import random
import numpy as np

from torch import nn
from torch.utils.data import DataLoader, Subset
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

if not os.path.exists('aux_files'):
    import build_embeddings_imdb
from SA_model import SentimentAnalysis
from data_cluster_seg import Data_infor
import argparse

SEED = 1234
random.seed(SEED)

def str2bool(string):
    if isinstance(string, bool):
        return string
    if string.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif string.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')
 
parser=argparse.ArgumentParser(
        description='LSTM Model Training')

parser.add_argument('--learning_rate', 
                    help = 'Learning rate', 
                    type = float,
                    default = 0.0005)
parser.add_argument('--nlayer',
                    help='The number of layers of LSTM',
                    type = int,
                    default = 2)
parser.add_argument('--bidirection',
                    help = 'Bidirectional LSTM',
                    type = str2bool,
                    default = True)
parser.add_argument('--kept_prob',
                    help = 'Probability to keep',
                    type = float,
                    default = 0.73)
parser.add_argument('--max_len',
                    help = 'Maximum length of sentence',
                    type = int,
                    default = 100) 
parser.add_argument('--save_path',
                    help = 'Save path',
                    default = '/content/drive/My Drive/Master_Final_Project/Genetic_attack/Code/nlp_adversarial_example_master_pytorch')

def train():
    args  = parser.parse_args()
    learning_rate = args.learning_rate
    nlayer = args.nlayer
    bidirection = args.bidirection
    save_path = args.save_path
    kept_prob = args.kept_prob
    
    MAX_VOCAB_SIZE = 50000
    with open(('aux_files/dataset_%d.pkl' %MAX_VOCAB_SIZE), 'rb') as f:
        dataset = pickle.load(f)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    embedding_matrix = np.load('aux_files/embeddings_glove_%d.npy' %(MAX_VOCAB_SIZE))
    embedding_matrix = torch.tensor(embedding_matrix.T).to(device)
    
    # pytorch
    max_len = args.max_len
    padded_train_raw = pad_sequences(dataset.train_seqs2, maxlen = max_len, padding = 'post')
    padded_test_raw = pad_sequences(dataset.test_seqs2, maxlen = max_len, padding = 'post')

    # TrainSet
    data_set_train = Data_infor(padded_train_raw, dataset.train_y)
    num_train = len(data_set_train)
    indx = list(range(num_train))
    all_train_set = Subset(data_set_train, indx)
#    train_indx = random.sample(indx, int(num_train*0.8))
#    vali_indx = [i for i in indx if i not in train_indx]
#    train_set = Subset(data_set_train, train_indx)
#    vali_set = Subset(data_set_train, vali_indx)
    
    
    # TestSet
    data_set_test = Data_infor(padded_test_raw, dataset.test_y)
    num_test = len(data_set_test)
    indx = list(range(num_test))
    # indx = random.sample(indx, SAMPLE_SIZE)
    test_set = Subset(data_set_test, indx)
    
    batch_size = 64
    hidden_size = 128
    all_train_loader = DataLoader(all_train_set, batch_size = batch_size, shuffle=True)
#    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle=True)
#    vali_loader = DataLoader(vali_set, batch_size = len(vali_indx)//batch_size)
    test_loader = DataLoader(test_set, batch_size = int(num_test/10), shuffle = True)
    best_save_path = os.path.join(save_path, 'best_lstm_'+str(kept_prob)+'_'+str(learning_rate)+'_test2')

    rnn = SentimentAnalysis(batch_size, embedding_matrix, hidden_size, kept_prob, nlayer, bidirection)
    rnn = rnn.to(device)
    # class my_loss(nn.Module):
    #   def __init__(self):
    #     super().__init__()
    #     self.relu = nn.ReLU()
          
    #   def forward(self, x, y):
    #     loss = torch.mean((1-y)*x+torch.log(1+torch.exp(-abs(x)))+self.relu(-x))
    #     return loss
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(rnn.parameters(), lr = learning_rate)
    # optimiser = torch.optim.SGD(rnn.parameters(), lr = learning_rate)
    
    epoches = 20
    best_epoch = 0
    best_acc = 0
    patience = 15
    
    for epoch in range(epoches):
      test_pred = torch.tensor([])
      test_targets = torch.tensor([])
      train_pred = torch.tensor([])
      train_targets = torch.tensor([])
      test_loss = []
      train_loss = []
    
      rnn.train()
      for batch_index, (seqs, length, target) in enumerate(all_train_loader):
        
        seqs = seqs.type(torch.LongTensor)
        len_order = torch.argsort(length, descending = True)
        length = length[len_order]
        seqs = seqs[len_order]
        target = target[len_order].type(torch.LongTensor)
        optimiser.zero_grad()
        seqs, target, length = seqs.to(device), target.to(device), length.to(device)
        output, pred_out = rnn(seqs, length, True)
        loss = criterion(output, target)
        loss.backward()
        optimiser.step()
    
        train_pred = torch.cat((train_pred, pred_out.type(torch.float).cpu()), dim = 0)
        train_targets = torch.cat((train_targets, target.type(torch.float).cpu()))
        train_loss.append(loss)
        
        if batch_index % 100 == 0:
          print('Train Batch:{}, Train Loss:{:.4f}.'.format(batch_index, loss.item()))
      train_accuracy = rnn.evaluate_accuracy(train_pred.detach().numpy(), train_targets.detach().numpy())
      print('Epoch:{}, Train Accuracy:{:.4f}, Train Mean loss:{:.4f}.'.format(epoch, train_accuracy, sum(train_loss)/len(train_loss)))
    
    
      rnn.eval()
      with torch.no_grad():
        for batch_index, (seqs, length, target) in enumerate(test_loader):
          
          seqs = seqs.type(torch.LongTensor)
          len_order = torch.argsort(length, descending = True)
          length = length[len_order]
          seqs = seqs[len_order]
          target = target[len_order].type(torch.LongTensor)
          seqs, target, length = seqs.to(device), target.to(device), length.to(device)
          output, pred_out = rnn(seqs, length, False)
          test_pred = torch.cat((test_pred, pred_out.type(torch.float).cpu()), dim = 0)
          test_targets = torch.cat((test_targets, target.type(torch.float).cpu()))
          loss = criterion(output, target)
          test_loss.append(loss.item())
          if batch_index % 100 == 0:
            print('Vali Batch:{}, Validation Loss:{:.4f}.'.format(batch_index, loss.item()))
        accuracy = rnn.evaluate_accuracy(test_pred.numpy(), test_targets.numpy())
        print('Epoch:{}, Vali Accuracy:{:.4f}, Vali Mean loss:{:.4f}.'.format(epoch, accuracy, sum(test_loss)/len(test_loss)))
        print('\n\n')
        # # best save
        # if accuracy > best_acc:
        #   best_acc = accuracy
        #   best_epoch = epoch
        #   torch.save(rnn.state_dict(), best_save_path)
        # # early stop
        # if epoch-best_epoch >=patience:
        #   print('Early stopping')
        #   print('Best epoch: {}, Best accuracy: {:.4f}.'.format(best_epoch, best_acc))
        #   break
    torch.save(rnn.state_dict(), best_save_path)
    rnn.load_state_dict(torch.load(best_save_path))
    rnn.to(device)
    rnn.eval()
    test_pred = torch.tensor([])
    test_targets = torch.tensor([])
    test_loss = []
    with torch.no_grad():
      for batch_index, (seqs, length, target) in enumerate(test_loader):
        
        seqs = seqs.type(torch.LongTensor)
        len_order = torch.argsort(length, descending = True)
        length = length[len_order]
        seqs = seqs[len_order]
        target = target[len_order]
        seqs, target, length = seqs.to(device), target.to(device), length.to(device)
        output, pred_out = rnn(seqs, length, False)
        test_pred = torch.cat((test_pred, pred_out.type(torch.float).cpu()), dim = 0)
        test_targets = torch.cat((test_targets, target.type(torch.float).cpu()))
        loss = criterion(output, target)
        test_loss.append(loss.item())
    
      accuracy = rnn.evaluate_accuracy(test_pred.numpy(), test_targets.numpy())
    print('Test Accuracy:{:.4f}, Test Mean loss:{:.4f}.'.format(accuracy, sum(test_loss)/len(test_loss)))
   
    
    
if __name__ == '__main__':
    train()
