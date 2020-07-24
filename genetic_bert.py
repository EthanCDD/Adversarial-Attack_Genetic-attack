# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 11:26:28 2020

@author: 13758
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 17:34:57 2020

@author: 13758
"""

import torch
import numpy as np
import glove_utils


class GeneticAttack_pytorch(object):
  def __init__(self, model, batch_model, neighbour_model, compute_dis,
               lm_model, tokenizer, max_iters, dataset,
               pop_size, n1, n2, n_prefix, n_suffix,
               use_lm = True, use_suffix = False):
    self.model = model
    self.batch_model = batch_model
    self.neighbour_model = neighbour_model
    self.model.eval()
    self.batch_model.eval()
    self.neighbour_model.eval()
    self.compute_dist = compute_dis
    self.pop_size = pop_size
    self.top_n1 = n1
    self.top_n2 = n2
    self.n_prefix = n_prefix
    self.n_suffix = n_suffix
    self.use_lm = use_lm
    self.tokenizer = tokenizer
    self.use_suffix = use_suffix
    self.w_i_dict = dataset.dict
    self.i_w_dict = dataset.inv_dict
    self.dataset = dataset
    self.lm = lm_model
    self.temp = 0.001
    self.max_iters = max_iters
    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    

  def attack(self, seq, target, l, max_change = 0.5):
    seq = seq.cpu().detach().numpy().squeeze()  #'''label of change; convert'''
    seq_adv = seq.copy()
    seq_len = np.sum(np.sign(seq))
    l = l.cpu()
    # print(self.tokenizer.convert_ids_to_tokens(seq.tolist()))
    # To calculate the sampling probability 
    tmp = [glove_utils.pick_most_similar_words(self.compute_dist(self.dataset.dict[self.tokenizer.convert_ids_to_tokens([seq[i]])[0]]), ret_count = 50, threshold = 0.5) if self.tokenizer.convert_ids_to_tokens([seq[i]])[0] in self.dataset.dict else ([], []) for i in range(l)]
    neighbour_list = [t[0] for t in tmp]
    neighbour_dist = [t[1] for t in tmp]
    neighbour_len = [len(i) for i in neighbour_list]
    for i in range(seq_len):
      if self.tokenizer.convert_ids_to_tokens([seq[i]])[0] in self.dataset.dict and self.dataset.dict[self.tokenizer.convert_ids_to_tokens([seq[i]])[0]] < 27:
        neighbour_len[i] = 0
    prob_select = neighbour_len/np.sum(neighbour_len)
    # print(prob_select)
    tmp = [glove_utils.pick_most_similar_words(
        self.compute_dist(self.dataset.dict[self.tokenizer.convert_ids_to_tokens([seq[i]])[0]]), self.top_n1, 0.5
    ) if self.tokenizer.convert_ids_to_tokens([seq[i]])[0] in self.dataset.dict else ([], []) for i in range(l)]
    neighbour_list = [t[0] for t in tmp]
    neighbour_dist = [t[1] for t in tmp]
    pop = [self.perturb(seq_adv, seq, neighbour_list, neighbour_dist, prob_select, seq_len, target, l) for _ in range(self.pop_size)]

    l_tensor = l*torch.ones([len(pop)]).type(torch.LongTensor)
    pop_np = np.expand_dims(pop[0], 0)
    for p in pop[1:]:
      pop_np = np.concatenate((pop_np, np.expand_dims(p, 0)),0) 

    for i in range(self.max_iters):
      pop_tensor = torch.tensor(pop_np).type(torch.LongTensor).to(self.device)
      l_tensor = l_tensor.to(self.device)
      self.batch_model.eval()
      with torch.no_grad():
        pop_preds = self.batch_model.pred(pop_tensor, l_tensor, False)[1].cpu().detach().numpy()
      
      pop_scores = pop_preds[:, target]
      print('\t\t', i, ' -- ', np.max(pop_scores))
      pop_ranks = np.argsort(pop_scores)[::-1]
      top_attack = pop_ranks[0]
      ampl = pop_scores / self.temp
      # print(ampl)
      covariance = np.cov(ampl)
#            print(covariance)
      if covariance>10e-4:
        mean = np.mean(ampl)
        # print(mean)
        ampl_update = (ampl-mean)/np.sqrt(covariance+0.001)
        # print(ampl_update)
        logits = np.exp(ampl_update)
      else:
        if np.max(ampl)>100:
          ampl = ampl/(np.max(ampl)/5)
        
        logits = np.exp(ampl)
      # logits = np.exp(ampl)
      select_probs = logits/np.sum(logits)
    
      if np.argmax(pop_preds[top_attack, :]) == target:
        print('Success and score: {:.4f}'.format(pop_scores[top_attack]))
        return pop[top_attack]
      
      elite = [pop[top_attack]]  # elite
      # print(select_probs.shape)
      parent1_idx = np.random.choice(
          self.pop_size, size=self.pop_size-1, p=select_probs)
      parent2_idx = np.random.choice(
          self.pop_size, size=self.pop_size-1, p=select_probs)
      
      childs = [self.crossover(pop[parent1_idx[i]],
                                pop[parent2_idx[i]])
                for i in range(self.pop_size-1)]
      childs = [self.perturb(
          x, seq, neighbour_list, neighbour_dist, prob_select, seq_len, target, l) for x in childs]
      pop = elite + childs
      pop_np = np.expand_dims(pop[0], 0)
      for p in pop[1:]:
        pop_np = np.concatenate((pop_np, np.expand_dims(p, 0)),0)

    return None
    
  def perturb(self, seq_cur, seq, neighbour_list, neighbour_dist, prob_select, seq_len, target, l):
    prob_len = len(prob_select)
    rand_idx = np.random.choice(prob_len, 1, p = prob_select)[0]
    while seq_cur[rand_idx] != seq[rand_idx] and np.sum(seq != seq_cur)<np.sum(np.sign(prob_select)):
      rand_idx = np.random.choice(prob_len, 1, p = prob_select)[0]
    
    replace_list = neighbour_list[rand_idx]
    # print(replace_list)
    if len(replace_list) < self.top_n1:
      replace_list = np.concatenate((replace_list, np.zeros(self.top_n1 - replace_list.shape[0])))
    return self.select_best_replacement(rand_idx, seq_cur, seq, target, l, replace_list)
  
  def replace(self, seq_cur, loc, w):
    seq_new = seq_cur.copy()
    seq_new[loc] = self.tokenizer.convert_tokens_to_ids(self.dataset.inv_dict[w])
    return seq_new


  def select_best_replacement(self, loc, seq_cur, seq, target, l, replace_list):
    new_seq_list = [self.replace(seq_cur, loc, w) if seq[loc]!=w and w != 0 else seq_cur for w in replace_list]
    # print('-------------------------')
    # print([self.tokenizer.convert_ids_to_tokens(seq) for seq in new_seq_list])
    # print('-------------------------')
    l_seq_list = len(new_seq_list)
    new_seq_list_tensor = torch.tensor(new_seq_list).type(torch.LongTensor).to(self.device)
    l_tensor = l*torch.ones([l_seq_list]).type(torch.LongTensor)
    l_tensor = l_tensor.to(self.device)
    self.neighbour_model.eval()
    with torch.no_grad():
      new_seq_preds = self.neighbour_model.pred(new_seq_list_tensor, l_tensor, False)[1].cpu().detach().numpy()
    # print(new_seq_preds)
    # print(target)
    new_seq_scores = new_seq_preds[:, target]
    seq_tensor = torch.tensor(np.expand_dims(seq_cur, axis = 0)).type(torch.LongTensor).to(self.device)
    l_tensor = l.to(self.device)
    self.model.eval()
    with torch.no_grad():
      orig_score = self.model.pred(seq_tensor, l_tensor, False)[1].cpu().detach().numpy()[0, target]
    new_seq_scores -= orig_score
    
    new_seq_scores[self.top_n1:] = -10000000
    
    if self.use_lm:
      prefix = ['']
      suffix = ['']
      if loc > 0 and loc<=self.n_prefix:
        prefix = [self.tokenizer.convert_ids_to_tokens([seq_cur[loc-i-1]])[0] for i in range(0, int(loc)-1)[::-1]]
      elif loc>self.n_prefix:
        prefix = [self.tokenizer.convert_ids_to_tokens([seq_cur[loc-i-1]])[0] for i in range(self.n_prefix)[::-1]]


#      orig_word = self.tokenizer.convert_ids_to_tokens([seq[loc]])[0]
      if self.use_suffix and loc < l-self.n_suffix and seq_cur[loc+self.n_suffix+1]!=0:
        suffix = [self.tokenizer.convert_ids_to_tokens([seq_cur[loc+i]])[0] for i in range(1,self.n_suffix+1)]
      elif self.use_suffix and loc < l:
        suffix = [self.tokenizer.convert_ids_to_tokens([seq_cur[loc+i]])[0] for i in range(1,l-loc-1)]
#      print(orig_word, [self.dataset.inv_dict[w] for w in replace_list[:self.top_n1] if w in self.dataset.inv_dict])
      # print(prefix, suffix)
      word_list = [prefix+[self.dataset.inv_dict[w]]+suffix if w in self.dataset.inv_dict else prefix+['UNK']+suffix for w in replace_list[:self.top_n1]]
#      replace_words_orig = [self.dataset.inv_dict[w] if w in self.dataset.inv_dict else 'UNK' for w in replace_list[:self.top_n1]] + [orig_word]
      # print(word_list)
      replace_words_scores = self.lm.get_probs(word_list)
        
      new_words_scores = np.array(replace_words_scores)
      rank_replaces_by_lm = np.argsort(new_words_scores)
      filtered_words_idx = rank_replaces_by_lm[self.top_n2:]

      new_seq_scores[filtered_words_idx] = -10000000

    if np.max(new_seq_scores)>0:    
      return new_seq_list[np.argsort(new_seq_scores)[-1]]
    return seq_cur

  def crossover(self, seq1, seq2):
    seq_new = seq1.copy()
    for i in range(len(seq1)):
        if np.random.uniform() < 0.5:
            seq_new[i] = seq2[i]
    return seq_new