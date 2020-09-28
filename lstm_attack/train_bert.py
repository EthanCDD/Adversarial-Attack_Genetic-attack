# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 15:25:42 2020

@author: 13758
"""

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
from torch.utils.data import Dataset, Subset, DataLoader
from tensorflow.keras.preprocessing.sequence import pad_sequences

import data_utils
import glove_utils
#from goog_lm import LM
import lm_data_utils
import lm_utils

if not os.path.exists('aux_files'):
    import build_embeddings
from data_cluster_seg import Data_infor
from compute_dist import compute_dis
from data_sampler import data_infor
from pre_processing import pre_processing
from transformers import BertModel, BertTokenizer
from model_lstm_bert import bert_lstm

from genetic_bert import GeneticAttack_pytorch
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
parser.add_argument('--tokenizer',
                    help = 'Pre-processing tokenizer',
                    default = 'bert')

parser.add_argument('--file_path',
                    help = 'Save path',
                    default = '/lustre/scratch/scratch/ucabdc3/lstm_attack')#
#parser.add_argument('--file_path',
#                   help = 'File path',
#                   default = '/content/drive/My Drive/Master_Final_Project/Genetic_attack/Code/nlp_adversarial_example_master_pytorch')
def data_loading(test_text, test_target, SAMPLE_SIZE):
    
    dataset = data_infor(test_text, test_target)
    len_test = len(dataset)
    indx = list(range(len_test))
    all_test_data = Subset(dataset, indx)
    indx = random.sample(indx, SAMPLE_SIZE)
    test_data = Subset(dataset, indx)
    return test_data, all_test_data



def run():
    args = parser.parse_args()
    data = args.data
    nlayer = args.nlayer
    model_path = os.path.join(file_path, 'bert_lstm_attack/model_params/best_bert_0.7_0.001')  
    file_path = os.path.join(args.file_path, 'lstm_attack') #'/content/drive/My Drive/Master_Final_Project/Genetic_attack/Code/nlp_adversarial_example_master_pytorch/glove.840B.300d.txt'#'/lustre/scratch/scratch/ucabdc3/lstm_attack'
    
    save_path = os.path.join(file_path, 'results')

    MAX_VOCAB_SIZE = 50000
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(file_path, 'dataset_%d.pkl' %MAX_VOCAB_SIZE), 'rb') as f:
        dataset = pickle.load(f)
        
    with open('aux_files/dataset_%d.pkl' %MAX_VOCAB_SIZE, 'rb') as f:
        dataset = pickle.load(f)

    
#    skip_list = np.load('aux_files/missed_embeddings_counter_%d.npy' %MAX_VOCAB_SIZE)
    embedding_matrix = np.load('aux_files/embeddings_glove_%d.npy' %(MAX_VOCAB_SIZE))
    embedding_matrix = torch.tensor(embedding_matrix.T).to(device)
    dist = np.load(('aux_files/dist_counter_%d.npy' %(MAX_VOCAB_SIZE)))
    dist[0,:] = 100000
    dist[:,0] = 100000
#    goog_lm = LM()
    
    # pytorch
    max_len = 250
#    padded_train_raw = pad_sequences(dataset.train_seqs2, maxlen = max_len, padding = 'post')
#    padded_test_raw = pad_sequences(dataset.test_seqs2, maxlen = max_len, padding = 'post')
#    # TrainSet
#    data_set = Data_infor(padded_train_raw, dataset.train_y)
#    num_train = len(data_set)
#    indx = list(range(num_train))
#    train_set = Subset(data_set, indx)
    if data.lower() == 'imdb':
        data_path = 'aclImdb'
        
    bert = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    data_processed = pre_processing(data_path, MAX_VOCAB_SIZE)
    tokenizer_selection = args.tokenizer
    if tokenizer_selection.lower() != 'bert':
        data_processed.processing()
        train_sequences, test_sequences = data_processed.bert_indx(tokenizer)
        print('Self preprocessing')
    else:
        data_processed.bert_tokenize(tokenizer)
        train_sequences, test_sequences = data_processed.bert_indx(tokenizer)
        print('BERT tokenizer')
    train_text_init, test_text_init = data_processed.numerical(tokenizer, train_sequences, test_sequences, max_len = 250)
    
        
#    train_text = pad_sequences(train_text_init, maxlen = max_len, padding = 'post')
    test_text = pad_sequences(test_text_init, maxlen = max_len, padding = 'post')
#    train_target = data_processed.all_train_labels
    test_target = data_processed.all_test_labels
    SAMPLE_SIZE = args.sample_size
    test_data, all_test_data = data_loading(test_text, test_target, SAMPLE_SIZE)

    # TestSet
    batch_size = 1
    
#    data_set = Data_infor(padded_test_raw, dataset.test_y)
#    num_test = len(data_set)
#    indx = list(range(num_test))
#
##    all_test_set  = Subset(data_set, indx)
#    indx = random.sample(indx, SAMPLE_SIZE)
#    test_set = Subset(data_set, indx)
#    
#    test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = False)

    
    test_loader_bert = DataLoader(test_data, batch_size = batch_size, shuffle = False)
    all_test_loader_bert  = DataLoader(all_test_data, batch_size = 128, shuffle = True)
    
    
    
    lstm_size = 128
    rnn_state_save = model_path ##os.path.join(file_path,'best_bert_0.8_0.001_bert')
    model = bert_lstm(bert, 2, True, nlayer, lstm_size, True, 0.73)# batch_size=batch_size, embedding_matrix = embedding_matrix, hidden_size = lstm_size, kept_prob = 0.73, num_layers=2, bidirection=True)
    model.eval()
    model.load_state_dict(torch.load(rnn_state_save))
    model = model.to(device)

    model.eval()
    test_pred = torch.tensor([])
    test_targets = torch.tensor([])

    with torch.no_grad():
      for batch_index, (seqs, length, target) in enumerate(all_test_loader_bert):
        seqs = seqs.type(torch.LongTensor)
        len_order = torch.argsort(length, descending = True)
        length = length[len_order]
        seqs = seqs[len_order]
        target = target[len_order]
        seqs, target, length = seqs.to(device), target.to(device), length.to(device)

        output = model.pred(seqs, length)
        test_pred = torch.cat((test_pred, output.cpu()), dim = 0)
        test_targets = torch.cat((test_targets, target.type(torch.float).cpu()))

      accuracy = model.evaluate_accuracy(test_pred.numpy(), test_targets.numpy())
    print('Test Accuracy:{:.4f}.'.format(accuracy))
    np.save(os.path.join(save_path,'accuracy.npy'), np.array(accuracy))
    print('\n')
    n1 = 8
    n2 = 4
    pop_size = 60
    max_iters = 20
    n_prefix = 6
    n_suffix = 6
    batch_model = bert_lstm(bert, 2, True, nlayer, lstm_size, True, 0.73)#SentimentAnalysis(batch_size=pop_size, embedding_matrix = embedding_matrix, hidden_size = lstm_size, kept_prob = 0.73, num_layers=2, bidirection=True)
    
    batch_model.eval()
    batch_model.load_state_dict(torch.load(rnn_state_save))
    batch_model.to(device)
    
    neighbour_model = bert_lstm(bert, 2, True, nlayer, lstm_size, True, 0.73)#SentimentAnalysis(batch_size=batch_size, embedding_matrix = embedding_matrix, hidden_size = lstm_size, kept_prob = 0.73, num_layers=2, bidirection=True)
    
    neighbour_model.eval()
    neighbour_model.load_state_dict(torch.load(rnn_state_save))
    neighbour_model.to(device)
    lm_model = gpt_2_get_words_probs()
    ga_attack = GeneticAttack_pytorch(model, batch_model, neighbour_model, compute_dis,
               lm_model, tokenizer = tokenizer, max_iters = max_iters, dataset = dataset,
               pop_size = pop_size, n1 = n1, n2 = n2,n_prefix = n_prefix, 
               n_suffix = n_suffix, use_lm = True, use_suffix = True)
    
    
    TEST_SIZE = args.test_size
    order_pre = 0
    n = 0
    seq_success = []
    seq_orig = []
    seq_orig_label = []
    word_varied = []

    seq_success_path = os.path.join(save_path,'seq_success_perplexity_bert.npy')
    seq_orig_path = os.path.join(save_path,'seq_orig_perplexity_bert.npy')
    seq_orig_label_path = os.path.join(save_path,'seq_orig_label_perplexity_bert.npy')
    word_varied_path = os.path.join(save_path,'word_varied_perplexity_bert.npy')
    
    if order_pre != 0:
      seq_success = np.load(seq_success_path, allow_pickle = True).tolist()
      seq_orig = np.load(seq_orig_path).tolist()
      seq_orig_label = np.load(seq_orig_label_path).tolist()
      word_varied = np.load(word_varied_path, allow_pickle = True).tolist()
      n = len(seq_success)
# 传输cuda的顺序; SEQ to numpy
    for order, (seq, l, target) in enumerate(test_loader_bert):

      if order>=order_pre:
        print('Sequence number:{}'.format(order))
        seq_len = np.sum(np.sign(seq.numpy()))
        seq = seq.type(torch.LongTensor)
        seq, l = seq.to(device), l.to(device)
        model.eval()
        with torch.no_grad():
          orig_pred = np.argmax(model.pred(seq, l).cpu().detach().numpy())
        if orig_pred != target.numpy()[0]:
          print('Wrong original prediction')
          print('----------------------')
          continue
        if seq_len > 100:
          print('Sequence is too long')
          print('----------------------')
          continue
    
        print('Length of sentence: {}, Number of samples:{}'.format(l.item(), n+1))
        seq_orig.append(seq[0].cpu().detach().numpy())
        seq_orig_label.append(target.numpy()[0])
        target = int(1-target.numpy()[0])
        seq_success.append(ga_attack.attack(seq, target, l.type(torch.LongTensor)))
        
        if None not in np.array(seq_success[n]):
          w_be = [tokenizer.convert_ids_to_tokens([seq_orig[n][i]]) for i in list(np.where(seq_success[n] != seq_orig[n])[0])]
          w_to = [tokenizer.convert_ids_to_tokens([seq_success[n][i]]) for i in list(np.where(seq_success[n] != seq_orig[n])[0])]
          for i in range(len(w_be)):
            print('{} ----> {}'.format(w_be[i], w_to[i]))
          word_varied.append([w_be]+[w_to])
        else:
          print('Fail')
        print('----------------------')
        n += 1
        
        np.save(seq_success_path, np.array(seq_success))
        np.save(seq_orig_path, np.array(seq_orig))
        np.save(seq_orig_label_path, np.array(seq_orig_label))
        np.save(word_varied_path, np.array(word_varied, dtype=object))
        
        if n>TEST_SIZE:
          break 
        
    



if __name__ == '__main__':
    run()