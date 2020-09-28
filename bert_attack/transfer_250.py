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

import data_utils_imdb
import glove_utils
#from goog_lm import LM
import lm_data_utils
import lm_utils

if not os.path.exists('aux_files'):
  import build_embeddings_imdb
from data_cluster_seg import Data_infor
from compute_dist import compute_dis
from data_sampler import data_infor
from pre_processing_transfer import pre_processing
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

# parser.add_argument('--file_path',
#                     help = 'Save path',
#                     default = '/lustre/scratch/scratch/ucabdc3/lstm_attack')#
parser.add_argument('--file_path',
                   help = 'File path',
                   default = '/content/drive/My Drive/Master_Final_Project/Genetic_attack/Code/nlp_adversarial_example_master_pytorch')
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
    file_path = args.file_path
    save_path = os.path.join(file_path, 'model_params')
    MAX_VOCAB_SIZE = 50000
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    with open('aux_files_transfer/dataset_%d.pkl' %MAX_VOCAB_SIZE, 'rb') as f:
        dataset = pickle.load(f)


    #    skip_list = np.load('aux_files/missed_embeddings_counter_%d.npy' %MAX_VOCAB_SIZE)
    embedding_matrix = np.load('aux_files_transfer/embeddings_glove_%d.npy' %(MAX_VOCAB_SIZE))
    embedding_matrix = torch.tensor(embedding_matrix.T).to(device)

    # pytorch
 
    if data.lower() == 'imdb':
        data_path = 'aclImdb'
        
    bert = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    with open('attack_results_final_250_AL_200.pkl', 'rb') as f:
        results= pickle.load(f)
    seqs = []
    lens = []
    tgts = []
    for i in range(len(results[1])):
        if np.array(results[1][i]).shape == ():
            continue
        seqs.append(' '.join([dataset.inv_dict[j] for j in results[1][i].tolist() if j!= 0]))
        lens.append(results[2][i])
        tgts.append(results[3][i])
    lens = torch.tensor(lens)
    tgts = torch.tensor(tgts)

    data_processed = pre_processing(seqs)
    tokenizer_select = args.tokenizer
    tokenizer_selection = tokenizer_select
    if tokenizer_selection.lower() != 'bert':
        data_processed.processing()
        train_sequences, test_sequences = data_processed.bert_indx(tokenizer)
        print('Self preprocessing')
    else:
        data_processed.bert_tokenize(tokenizer)
        test_sequences = data_processed.bert_indx(tokenizer)
        print('BERT tokenizer')
    test_text_init = data_processed.numerical(tokenizer,test_sequences)
        
    max_len = max([len(s) for s in test_text_init])
    test_text = pad_sequences(test_text_init, maxlen = max_len, padding = 'post')
    all_test_data = TensorDataset(torch.tensor(test_text), lens, tgts)
    all_test_loader_bert  = DataLoader(all_test_data, batch_size = 128, shuffle = True)



    lstm_size = 128
    rnn_state_save = os.path.join(save_path,'best_bert_0.7_0.001_bert_250')
    model = bert_lstm(bert, 2, False, nlayer, lstm_size, True, 0.7)# batch_size=batch_size, embedding_matrix = embedding_matrix, hidden_size = lstm_size, kept_prob = 0.73, num_layers=2, bidirection=True)
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

        output, pred_out = model.pred(seqs, length, False)
        test_pred = torch.cat((test_pred, pred_out.cpu()), dim = 0)
        test_targets = torch.cat((test_targets, target.type(torch.float).cpu()))

      accuracy = model.evaluate_accuracy(test_pred.numpy(), test_targets.numpy())
    print('Test Accuracy:{:.4f}.'.format(accuracy))



if __name__ == '__main__':
    run()