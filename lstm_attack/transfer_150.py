# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 18:51:19 2020

@author: 13758
"""

import os
import sys
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
from torch.utils.data import Dataset, Subset, DataLoader, TensorDataset
from tensorflow.keras.preprocessing.sequence import pad_sequences

import data_utils
import glove_utils
#from goog_lm import LM
import lm_data_utils
import lm_utils

if not os.path.exists('aux_files'):
    import build_embeddings
from compute_dist import compute_dis
from SA_model import SentimentAnalysis
from data_cluster_seg import Data_infor
from genetic_0003 import GeneticAttack_pytorch
from gpt_perplexity import gpt_2_get_words_probs
import argparse



SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def str2bool(string):
    if isinstance(string, bool):
        return string
    if string.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif string.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected')
    

parser = argparse.ArgumentParser(
        description = 'Adversarial examples generation/sentiment analysis'
        )

parser.add_argument('--nlayer',
                    help = 'The number of LSTM layers',
                    type = int,
                    default = 2)
parser.add_argument('--max_len',
                    help = 'Maximum attack length',
                    type = int,
                    default = 100)
parser.add_argument('--bidirection',
                    help = 'Bidirectional LSTM',
                    type = str2bool,
                    default = True)
parser.add_argument('--data',
                    help = 'The applied dataset',
                    default = 'IMDB')
parser.add_argument('--sample_size',
                    help = 'The number of examples to generate adversarial samples',
                    type = int,
                    default = 10000)
parser.add_argument('--test_size',
                    help = 'The number of tested examples',
                    type = int,
                    default = 1000)
parser.add_argument('--use_lm', 
                    help = 'Language model application',
                    type = str2bool,
                    default = True)
parser.add_argument('--file_path',
                    help = 'Save path',
                    default = '/lustre/scratch/scratch/ucabdc3/lstm_attack')
#parser.add_argument('--file_path',
#                    help = 'File path',
#                    default = '/content/drive/My Drive/Master_Final_Project/Genetic_attack/Code/nlp_adversarial_example_master_pytorch')

def run():
    args = parser.parse_args()
    nlayer = args.nlayer
    bidirection = args.bidirection
    file_path = args.file_path#'/content/drive/My Drive/Master_Final_Project/Genetic_attack/Code/nlp_adversarial_example_master_pytorch/glove.840B.300d.txt'#'/lustre/scratch/scratch/ucabdc3/lstm_attack'
    save_path = os.path.join(file_path, 'results')
    MAX_VOCAB_SIZE = 50000
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#    with open(os.path.join(file_path, 'dataset_%d.pkl' %MAX_VOCAB_SIZE), 'rb') as f:
#        dataset = pickle.load(f)
        
    with open('aux_files/dataset_%d.pkl' %MAX_VOCAB_SIZE, 'rb') as f:
        dataset = pickle.load(f)

        
#    skip_list = np.load('aux_files/missed_embeddings_counter_%d.npy' %MAX_VOCAB_SIZE)
    embedding_matrix = np.load('aux_files/embeddings_glove_%d.npy' %(MAX_VOCAB_SIZE))
    embedding_matrix = torch.tensor(embedding_matrix.T).to(device)
    dist = np.load(('aux_files/dist_counter_%d.npy' %(MAX_VOCAB_SIZE)))

    
#    goog_lm = LM()
    
    # pytorch
    max_len = args.max_len
#    padded_train_raw = pad_sequences(dataset.train_seqs2, maxlen = max_len, padding = 'post')
    padded_test_raw = pad_sequences(dataset.test_seqs2, maxlen = max_len, padding = 'post')
#    # TrainSet
#    data_set = Data_infor(padded_train_raw, dataset.train_y)
#    num_train = len(data_set)
#    indx = list(range(num_train))
#    train_set = Subset(data_set, indx)
    
    # TestSet
    batch_size = 1
    SAMPLE_SIZE = args.sample_size
    data_set = Data_infor(padded_test_raw, dataset.test_y)
    num_test = len(data_set)
    indx = list(range(num_test))
    
    all_test_set  = Subset(data_set, indx)
    #indx = random.sample(indx, SAMPLE_SIZE)
    with open('attack_results_final_150_AL_150.pkl', 'rb') as f:
        results= pickle.load(f)
    seqs = []
    lens = []
    tgts = []
    for i in range(len(results[1])):
        if np.array(results[1][i]).shape == ():
            continue
        seqs.append(results[1][i])
        lens.append(results[2][i])
        tgts.append(results[3][i])
    seqs = torch.tensor(seqs)
    lens = torch.tensor(lens)
    tgts = torch.tensor(tgts)
    test_set = TensorDataset(seqs, lens, tgts)
    all_test_loader  = DataLoader(test_set, batch_size = 128, shuffle = True)
    
    lstm_size = 128
    rnn_state_save = os.path.join(file_path,'best_lstm_0.7_0.001_150')

    model = SentimentAnalysis(batch_size=lstm_size, embedding_matrix = embedding_matrix, hidden_size = lstm_size, kept_prob = 0.7, num_layers=nlayer, bidirection=bidirection)
    
    model.load_state_dict(torch.load(rnn_state_save))
    model = model.to(device)
    
    
    model.eval()
    test_pred = torch.tensor([])
    test_targets = torch.tensor([])

    with torch.no_grad():
      for batch_index, (seqs, length, target) in enumerate(all_test_loader):
        seqs, target, length = seqs.to(device), target.to(device), length.to(device)
        seqs = seqs.type(torch.LongTensor)
        len_order = torch.argsort(length, descending = True)
        length = length[len_order]
        seqs = seqs[len_order]
        target = target[len_order]

        output, pred_out = model.pred(seqs, length, False)
        test_pred = torch.cat((test_pred, pred_out.cpu()), dim = 0)
        test_targets = torch.cat((test_targets, target.type(torch.float).cpu()))

      accuracy = model.evaluate_accuracy(test_pred.numpy(), test_targets.numpy())
    print('Test Accuracy:{:.4f}.'.format(accuracy))

if __name__ == '__main__':
    run()