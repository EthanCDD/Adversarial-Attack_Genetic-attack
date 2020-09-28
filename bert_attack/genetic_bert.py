
import torch
import numpy as np
import glove_utils
# from lm_scorer.models.auto import AutoLMScorer as LMScorer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
    self.temp = 0.0002
    self.max_iters = max_iters
    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # batch_size = 1
    # self.scorer = LMScorer.from_pretrained("gpt2", device=self.device, batch_size=batch_size)

    

  def attack(self, seq, target, l, max_change = 0.5):
    
    seq = seq.cpu().detach().numpy().squeeze()  #'''label of change; convert'''
    seq_orig, seq_orig_string, l_orig = self.orig_sentence(seq)
    
    # print(seq_orig)
    # seq_adv = seq.copy()
    # seq_len = np.sum(np.sign(seq))
    l = l.cpu()
    # print(self.tokenizer.convert_ids_to_tokens(seq.tolist()))
    # To calculate the sampling probability 
    tmp = [glove_utils.pick_most_similar_words(
            self.compute_dist(seq_orig[i]), 50, 0.5) for i in range(l_orig)]

    # tmp = [glove_utils.pick_most_similar_words(self.compute_dist(self.dataset.dict[self.tokenizer.convert_ids_to_tokens([seq[i]])[0]]), ret_count = 50, threshold = 0.5) if self.tokenizer.convert_ids_to_tokens([seq[i]])[0] in self.dataset.dict else ([], []) for i in range(l)]
    neighbour_list = [t[0] for t in tmp]
    neighbour_dist = [t[1] for t in tmp]
    neighbour_len = [len(i) for i in neighbour_list]
    for i in range(l_orig):
      if (seq_orig[i] < 27):
        # To prevent replacement of words like 'the', 'a', 'of', etc.
        neighbour_len[i] = 0
    prob_select = neighbour_len/np.sum(neighbour_len)
    # print(prob_select)
    # tmp = [glove_utils.pick_most_similar_words(
    #     self.compute_dist(self.dataset.dict[self.tokenizer.convert_ids_to_tokens([seq[i]])[0]]), self.top_n1, 0.5
    # ) if self.tokenizer.convert_ids_to_tokens([seq[i]])[0] in self.dataset.dict else ([], []) for i in range(l)]
    tmp = [glove_utils.pick_most_similar_words(
            self.compute_dist(seq_orig[i]), self.top_n1, 0.5) for i in range(l_orig)]

    neighbour_list = [t[0] for t in tmp]
    neighbour_dist = [t[1] for t in tmp]
    # print('synonyms')
    # print(tmp)
    # print([[self.dataset.inv_dict[j] for j in i if j in self.dataset.inv_dict] for i in neighbour_list])
    seq_adv = seq_orig_string.copy()
    # pop = [self.perturb(seq_adv, seq, seq_orig, l_orig, neighbour_list, neighbour_dist, prob_select, seq_len, target, l) for _ in range(self.pop_size)]
    pop = [self.perturb(seq_adv, seq_orig_string, l_orig, neighbour_list, neighbour_dist, prob_select, target, l) for _ in range(self.pop_size)]


    l_tensor = torch.ones([len(pop)]).type(torch.LongTensor)
    pop_np = [[self.tokenizer.cls_token_id] + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' '.join(pop[0]).strip())) + [self.tokenizer.sep_token_id]]
    l_tensor[0] = len(pop_np[0])
    # print(l_tensor)
    for p in range(1, len(pop)):
      token_ids = [self.tokenizer.cls_token_id] + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' '.join(pop[p]).strip())) + [self.tokenizer.sep_token_id]
      pop_np.append(token_ids) 
      l_tensor[p] = len(token_ids)
    l_max = torch.max(l_tensor)

    # print(l_max, l_tensor, len(pop_np))
    pop_np = pad_sequences(pop_np, maxlen = l_max.item(), padding = 'post')
    pop_tensor = torch.tensor(pop_np)

    # print(torch.tensor(pop_np))
    sort = torch.sort(l_tensor, descending = True)[1]
    # print(len(sort), sort)
    pop_tensor = pop_tensor[sort]
    l_tensor = l_tensor[sort]
    pop = np.array(pop)[sort].tolist()
    # print(l_tensor)
    for i in range(self.max_iters):

      pop_tensor = pop_tensor.type(torch.LongTensor).to(self.device)
      l_tensor = l_tensor.to(self.device)
      # print('pop_tensor:',pop_tensor)
      # print(pop_tensor.shape)
      # print(l_tensor)
      self.batch_model.eval()
      with torch.no_grad():
        pop_preds = self.batch_model.pred(pop_tensor, l_tensor, False)[1].cpu().detach().numpy()
      # print(sort)
      # print(pop_preds)
      # print(pop_tensor)
      pop_scores = pop_preds[:, target]
      print('\t\t', i, ' -- ', np.max(pop_scores))
      pop_ranks = np.argsort(pop_scores)[::-1]
      # print(l_tensor)
      # print(pop_ranks)
      top_attack = pop_ranks[0]
      # print(top_attack)
      ampl = pop_scores / self.temp
      # print(ampl)
      covariance = np.cov(ampl)
      # print('pop:', pop)
      print(covariance)
      if covariance>10e-6:
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
      # print('prob:', select_probs)
      # print([self.tokenizer.convert_ids_to_tokens([i]) for i in pop_np[top_attack]])
      if np.argmax(pop_preds[top_attack, :]) == target:
        print('Success and score: {:.4f}'.format(pop_scores[top_attack]))

        print(seq_orig_string)
        print(pop[top_attack])

        return pop[top_attack], seq_orig_string
      
      # for i in pop:
      #   print(i)
      #   print('\t')

      elite = [pop[top_attack]]  # elite
      # print(elite)

      # print(select_probs.shape)
      parent1_idx = np.random.choice(
          self.pop_size, size=self.pop_size-1, p=select_probs)
      parent2_idx = np.random.choice(
          self.pop_size, size=self.pop_size-1, p=select_probs)
      
      childs = [self.crossover(pop[parent1_idx[i]],
                                pop[parent2_idx[i]])
                for i in range(self.pop_size-1)]
      childs = [self.perturb(
           x, seq_orig_string, l_orig, neighbour_list, neighbour_dist, prob_select, target, l) for x in childs]
      # print(childs)
      pop = elite+childs
      # print(len(pop))
      # print('pop:', pop)      
      l_tensor = torch.ones([len(pop)]).type(torch.LongTensor)
      pop_np = [[self.tokenizer.cls_token_id] + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' '.join(pop[0]).strip())) + [self.tokenizer.sep_token_id]]
      l_tensor[0] = len(pop_np[0])
      # print(pop_np)
      # print(l_tensor)
      # print(pop_np)
      for p in range(1, len(pop)):
        token_ids = [self.tokenizer.cls_token_id] + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' '.join(pop[p]).strip())) + [self.tokenizer.sep_token_id]
        pop_np.append(token_ids) 
        l_tensor[p] = len(token_ids)
      
      # print(l_tensor)
      # print(pop_np)
      l_max = torch.max(l_tensor)
      pop_np = pad_sequences(pop_np, maxlen = l_max.item(), padding = 'post')
      pop_tensor = torch.tensor(pop_np)

      # print(torch.tensor(pop_np))
      sort = torch.sort(l_tensor, descending = True)[1]
      # print(len(sort), sort)
      pop_tensor = pop_tensor[sort]
      l_tensor = l_tensor[sort]
      pop = np.array(pop)[sort].tolist()
      # print(np.array(pop).shape)

      # pop_np = np.expand_dims(pop[0], 0)
      # for p in pop[1:]:
      #   pop_np = np.concatenate((pop_np, np.expand_dims(p, 0)),0)

    return None, seq_orig

  def orig_sentence(self, seq):
    seq_orig=[]
    for i in seq:
      w = self.tokenizer.convert_ids_to_tokens([i])[0]
      # print(w)
      if i == 0:
        break
      if len(w)>1 and w[:2] == '##' and seq_orig !=[]:
        # print(seq_orig, w[2:])
        seq_orig[-1] = seq_orig[-1]+w[2:]
      else:
        seq_orig.append(w)
    seq_string = seq_orig.copy()[1:-1]
    l_orig = len(seq_orig)
    seq_orig = [self.dataset.dict[seq_orig[i]] if seq_orig[i] in self.dataset.dict else self.dataset.dict['UNK'] for i in range(l_orig)][1:-1]
    l_orig -= 2
    return seq_orig, seq_string, l_orig

  def perturb(self, seq_cur, seq, l_orig, neighbour_list, neighbour_dist, prob_select, target, l):
    # print('perturb:',seq)
    prob_len = len(prob_select)
    rand_idx = np.random.choice(prob_len, 1, p = prob_select)[0]
    while seq_cur[rand_idx].strip() != seq[rand_idx].strip() and np.sum(seq != seq_cur)<np.sum(np.sign(prob_select)):
      # print(seq_cur[rand_idx], seq[rand_idx], np.sum(seq != seq_cur))
      rand_idx = np.random.choice(prob_len, 1, p = prob_select)[0]
    
    replace_list = neighbour_list[rand_idx]
    # print(replace_list)
    if len(replace_list) < self.top_n1:
      replace_list = np.concatenate((replace_list, np.zeros(self.top_n1 - replace_list.shape[0])))
    return self.select_best_replacement(rand_idx, seq_cur, seq, l_orig, target, replace_list, l)
  
  def replace(self, seq_str, loc, w):
    seq_new = seq_str.copy()
    seq_new[loc] = self.dataset.inv_dict[w]#self.dataset.inv_dict[w]#self.tokenizer.convert_tokens_to_ids()
    seq_bert = [self.tokenizer.cls_token_id] + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' '.join(seq_new).strip())) + [self.tokenizer.sep_token_id]
    l_bert = len(seq_bert)
    # print(l_bert)
    return seq_bert, seq_new, l_bert

  # def word_pre(self, w, i, loc, seq_cur):
  #   if len(w)<2 or w[:2] != '##':
  #     return
  #   part = self.tokenizer.convert_ids_to_tokens([seq_cur[loc-i-1]])[0]
  #   return self.word_pre(part+w[2:], i+1, seq_cur)
  

  def select_best_replacement(self, pos, seq_cur, seq, l_orig, target, replace_list, l):

    infor_list = [self.replace(seq_cur, pos, w) if w != 0 and seq[pos].strip()!=self.dataset.inv_dict[w]  else \
                  (([self.tokenizer.cls_token_id] + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' '.join(seq).strip()))\
                    + [self.tokenizer.sep_token_id]), seq_cur, l) for w in replace_list]

    n_seq_list = len(infor_list)
    new_seq_list = []
    cur_seq_list = []
    l_bert_list = []
    for i in range(n_seq_list):
      new_seq_list.append(infor_list[i][0])
      cur_seq_list.append(infor_list[i][1])
      l_bert_list.append(infor_list[i][2])
    # print(cur_seq_list)
    # print(l_bert_list)
    # print([self.tokenizer.convert_ids_to_tokens([i]) for i in new_seq_list[0]])
    l_bert_list = torch.tensor(l_bert_list)
    sort = torch.argsort(l_bert_list, descending = True)

    l_max_bert = torch.max(l_bert_list)
    new_seq_list = pad_sequences(new_seq_list, maxlen = l_max_bert, padding = 'post')
    new_seq_list_tensor = torch.tensor(new_seq_list)[sort].type(torch.LongTensor).to(self.device)
    replace_list = replace_list[sort]
    # print('replace_list:', replace_list)

    l_tensor = l_bert_list[sort].type(torch.LongTensor)
    l_tensor = l_tensor.to(self.device)
    # print(new_seq_list_tensor)
    self.neighbour_model.eval()
    with torch.no_grad():
      new_seq_preds = self.neighbour_model.pred(new_seq_list_tensor, l_tensor, False)[1].cpu().detach().numpy()
    # print(new_seq_preds)
    # print(target)
    new_seq_scores = new_seq_preds[:, target]
    # print(new_seq_scores)
    # print(' '.join([self.dataset.inv_dict[i] if i!=50000 else '[UNK]' for i in seq_cur]).strip())
    seq_np = np.expand_dims([self.tokenizer.cls_token_id] + self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(' '.join(seq_cur).strip())) + [self.tokenizer.sep_token_id], axis = 0)
    seq_tensor = torch.tensor(seq_np).type(torch.LongTensor).to(self.device) #torch.tensor(np.expand_dims(seq_cur, axis = 0)).type(torch.LongTensor).to(self.device)
    # print([self.tokenizer.convert_ids_to_tokens([i]) for i in seq_tensor[0]])
    l_tensor = torch.tensor([seq_tensor.shape[1]]).to(self.device)
    # print(seq_tensor)
    self.model.eval()
    with torch.no_grad():
      orig_score = self.model.pred(seq_tensor, l_tensor, False)[1].cpu().detach().numpy()[0, target]
    new_seq_scores -= orig_score
    # print(new_seq_scores)
    
    new_seq_scores[self.top_n1:] = -10000000
    # print(new_seq_scores)
    if self.use_lm:
      prefix = ['']
      suffix = ['']
      if pos > 0 and pos<=self.n_prefix:
        prefix = [seq_cur[pos-i-1] for i in range(int(pos))[::-1]]
      elif pos>self.n_prefix:
        prefix = [seq_cur[pos-i-1] for i in range(self.n_prefix)[::-1]]

#      orig_word = self.i_w_dict[seq[loc]]
      if self.use_suffix and pos < l_orig-self.n_suffix:
        suffix = [seq_cur[pos+i] for i in range(1,self.n_suffix+1)]
      elif self.use_suffix and pos < l_orig:
        suffix = [seq_cur[pos+i] for i in range(1,l_orig-pos)]

#     if self.use_lm:
#       prefix = ['']
#       suffix = ['']
#       if loc > 0 and loc<=self.n_prefix:
#         prefix = [self.tokenizer.convert_ids_to_tokens([seq_cur[loc-i-1]])[0] for i in range(0, int(loc)-1)[::-1]]
#       elif loc>self.n_prefix:
#         prefix = [self.tokenizer.convert_ids_to_tokens([seq_cur[loc-i-1]])[0] for i in range(self.n_prefix)[::-1]]


# #      orig_word = self.tokenizer.convert_ids_to_tokens([seq[loc]])[0]
#       if self.use_suffix and loc < l-self.n_suffix and seq_cur[loc+self.n_suffix+1]!=0:
#         suffix = [self.tokenizer.convert_ids_to_tokens([seq_cur[loc+i]])[0] for i in range(1,self.n_suffix+1)]
#       elif self.use_suffix and loc < l:
#         suffix = [self.tokenizer.convert_ids_to_tokens([seq_cur[loc+i]])[0] for i in range(1,l-loc-1)]

#     if self.use_lm:
#       prefix = ['']
#       suffix = ['']
#       print(loc)
#       if loc > 0 and loc<=self.n_prefix:
#         prefix = []
#         for i in range(0, int(loc)-1)[::-1]:
#           w = self.tokenizer.convert_ids_to_tokens([seq_cur[loc-i-1]])[0]
#           if len(w)>2:
#             if w[:2] == '##' and i != int(loc)-2:
#               print(w)
#               print(prefix)
#               w = prefix[-1]+w[2:]
#               prefix[-1] = w
            
#             elif w[:2] == '##' and i == int(loc)-2:
#               print(w)
#               print(prefix)
#               w = self.word_pre(w, i+1, loc, seq_cur)  
#               prefix.append(w)
#             else:
#               prefix.append(w)
#           else:
#             prefix.append(w)
#           print(w)
#           print('pre:',prefix)
          
#         # prefix = [self.tokenizer.convert_ids_to_tokens([seq_cur[loc-i-1]])[0] for i in range(0, int(loc)-1)[::-1]]
#       elif loc>self.n_prefix:
#         prefix = []
#         for i in range(self.n_prefix)[::-1]:
#           print(loc-i-1)
#           w = self.tokenizer.convert_ids_to_tokens([seq_cur[loc-i-1]])[0]
#           if len(w)>2:
#             print(i, int(loc)-2)
#             if w[:2] == '##' and i != self.n_prefix-1:
#               print(w)
#               print(prefix)
#               w = prefix[-1]+w[2:]
#               prefix[-1] = w
#             elif w[:2] == '##' and i == self.n_prefix-1:
#               print(w)
#               print(prefix)
#               w = self.word_pre(w, i+1, loc, seq_cur)  
#               prefix.append(w)
#             else:
#               prefix.append(w)
#           else:
#             prefix.append(w)
#           print(w)
#           print(prefix)
#         # prefix = [self.tokenizer.convert_ids_to_tokens([seq_cur[loc-i-1]])[0] for i in range(self.n_prefix)[::-1]]
#       print('prefix:', prefix)
#       print(loc+self.n_suffix)
# #      orig_word = self.tokenizer.convert_ids_to_tokens([seq[loc]])[0]
#       if self.use_suffix and loc < l-self.n_suffix-1 and seq_cur[loc+self.n_suffix]!=0:
#         suffix = []
#         for i in range(1,self.n_suffix+1):
#           print(loc+i)
#           w = self.tokenizer.convert_ids_to_tokens([seq_cur[loc+i]])[0]
#           if len(w)>2:
#             if w[:2] == '##' and i != 1:
#               print(w)
#               print(suffix)
#               w = suffix[-1]+w[2:]
#               suffix[-1] = w
#             elif w[:2] == '##' and i == 1:
#               print(w)
#               print(suffix)
#               w = self.word_pre(w, i+1, loc, seq_cur)  
#               suffix.append(w)
#             else:
#               suffix.append(w)
#           else:
#             suffix.append(w)
#           print(suffix)
#         # suffix = [self.tokenizer.convert_ids_to_tokens([seq_cur[loc+i]])[0] for i in range(1,self.n_suffix+1)]
#       elif self.use_suffix and loc < l:
#         suffix = []
#         for i in range(1,l-loc-1):
#           w = self.tokenizer.convert_ids_to_tokens([seq_cur[loc+i]])[0]
#           if len(w)>2:
#             if w[:2] == '##' and i != int(loc)+1:
#               print(w)
#               print(suffix)
#               w = suffix[-1]+w[2:]
#               suffix[-1] = w
#             elif w[:2] == '##' and i == int(loc)+1:
#               print(w)
#               print(suffix)
#               w = self.word_pre(w, i+1, loc, seq_cur)  
#               suffix.append(w)
#             else:
#               suffix.append(w)
#           else:
#             suffix.append(w)
      
#       print('suffix:', suffix)
        # suffix = [self.tokenizer.convert_ids_to_tokens([seq_cur[loc+i]])[0] for i in range(1,l-loc-1)]
#      print(orig_word, [self.dataset.inv_dict[w] for w in replace_list[:self.top_n1] if w in self.dataset.inv_dict])
      # print(prefix, suffix)
      word_list = [prefix+[self.dataset.inv_dict[w]]+suffix if w in self.dataset.inv_dict else prefix+['UNK']+suffix for w in replace_list]
#[prefix+[self.dataset.inv_dict[w]]+suffix if w in self.dataset.inv_dict else prefix+['UNK']+suffix for w in replace_list[:self.top_n1]]
#      replace_words_orig = [self.dataset.inv_dict[w] if w in self.dataset.inv_dict else 'UNK' for w in replace_list[:self.top_n1]] + [orig_word]
      # print(word_list)
      # print(word_list)
      # print('replace_list:', [self.dataset.inv_dict[i] if i in self.dataset.inv_dict else i for i in replace_list])

      # seqs = [self.seq_list(seq) for seq in word_list]
      # replace_words_scores = self.scorer.sentence_score(seqs, reduce = 'prod')
      # new_words_scores = np.array(replace_words_scores)
      # rank_replaces_by_lm = np.argsort(new_words_scores)[::-1]
      # # print(new_words_scores[rank_replaces_by_lm])
      # # print(rank_replaces_by_lm)


      replace_words_scores = self.lm.get_probs(word_list)
      new_words_scores = np.array(replace_words_scores)
      rank_replaces_by_lm = np.argsort(new_words_scores)


      filtered_words_idx = rank_replaces_by_lm[self.top_n2:]

      new_seq_scores[filtered_words_idx] = -10000000

    if np.max(new_seq_scores)>0:  
      # print([self.dataset.inv_dict[i] for i in cur_seq_list[np.argsort(new_seq_scores)[-1]]])  
      return cur_seq_list[np.argsort(new_seq_scores)[-1]]
    return seq_cur

  def seq_list(self, prefix):
      sentence = ''

      for word in prefix:
          sentence += word + ' '
      sentence = sentence.strip()
      return sentence

  def crossover(self, seq1, seq2):
    # print('seq1:', seq1)
    # print('seq2:', seq2)
    seq_new = seq1.copy()
    for i in range(len(seq1)):
        if np.random.uniform() < 0.5:
            seq_new[i] = seq2[i]
    return seq_new