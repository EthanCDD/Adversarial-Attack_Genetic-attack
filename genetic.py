import torch
import numpy as np
import glove_utils
import torch
from lm_scorer.models.auto import AutoLMScorer as LMScorer

class GeneticAttack_pytorch(object):
    def __init__(self, model, batch_model, neighbour_model, compute_dis,
               lm, max_iters, dataset,
               pop_size, n1, n2, n_prefix, n_suffix,
               use_lm = True, use_suffix = False):
#        self.dist_mat = dist_mat
        self.compute_dist = compute_dis
        self.dataset = dataset
        self.dict = self.dataset.dict
        self.inv_dict = self.dataset.inv_dict
#        self.skip_list = skip_list
        self.model = model
        self.batch_model = batch_model
        self.neighbour_model = neighbour_model
#        self.sess = sess
        self.n_prefix = n_prefix
        self.n_suffix = n_suffix
        self.max_iters = max_iters
        self.pop_size = pop_size
        self.lm = lm
        self.top_n = n1  # similar words
        self.top_n1 = n1
        self.top_n2 = n2
        self.use_lm = use_lm
        self.use_suffix = use_suffix
        self.temp = 0.0003
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        batch_size = 1
        self.scorer = LMScorer.from_pretrained("gpt2", device=self.device, batch_size=batch_size)

    def do_replace(self, x_cur, pos, new_word):
        x_new = x_cur.copy()
        x_new[pos] = new_word
        return x_new

    def select_best_replacement(self, pos, x_len, x_cur, x_orig, target, replace_list):
        """ Select the most effective replacement to word at pos (pos)
        in (x_cur) between the words in replace_list """
        new_x_list = [self.do_replace(
            x_cur, pos, w) if x_orig[pos] != w and w != 0 else x_cur for w in replace_list]
#        new_x_preds = self.neighbour_model.predict(
#            self.sess, np.array(new_x_list))
        l_seq_list = len(new_x_list)
        new_seq_list_tensor = torch.tensor(new_x_list).type(torch.LongTensor).to(self.device)
        l_tensor = x_len*torch.ones([l_seq_list])
        l_tensor = l_tensor.to(self.device)
        with torch.no_grad():
            new_x_preds = self.neighbour_model.pred(new_seq_list_tensor, l_tensor, False)[1].cpu().detach().numpy()

        # Keep only top_n
        # replace_list = replace_list[:self.top_n]
        #new_x_list = new_x_list[:self.top_n]
        #new_x_preds = new_x_preds[:self.top_n,:]
        # new_x_scores = new_x_preds[:, target]
        # orig_score = self.model.pred(
        #     self.sess, x_cur[np.newaxis, :])[1][0, target]
        
        new_x_scores = new_x_preds[:, target]
        seq_tensor = torch.tensor(np.expand_dims(x_cur, axis = 0)).type(torch.LongTensor).to(self.device)
        l_tensor = torch.tensor([x_len]).to(self.device)
        self.model.eval()
        with torch.no_grad():
          orig_score = self.model.pred(seq_tensor, l_tensor, False)[1].cpu().detach().numpy()[0, target]
        new_x_scores = new_x_scores - orig_score
        
        # Eliminate not that clsoe words
        new_x_scores[self.top_n:] = -10000000

#        if self.use_lm:
#            prefix = ""
#            suffix = None
#            if pos > 0:
#                prefix = self.dataset.inv_dict[x_cur[pos-1]]
#            #
#            orig_word = self.dataset.inv_dict[x_orig[pos]]
#            if self.use_suffix and pos < x_cur.shape[0]-1:
#                if (x_cur[pos+1] != 0):
#                    suffix = self.dataset.inv_dict[x_cur[pos+1]]
#            # print('** ', orig_word)
#            replace_words_and_orig = [
#                self.dataset.inv_dict[w] if w in self.dataset.inv_dict else 'UNK' for w in replace_list[:self.top_n]] + [orig_word]
#            # print(replace_words_and_orig)
#            replace_words_lm_scores = self.lm.get_words_probs(
#                prefix, replace_words_and_orig, suffix)
#            # print(replace_words_lm_scores)
#            # for i in range(len(replace_words_and_orig)):
#            #    print(replace_words_and_orig[i], ' -- ', replace_words_lm_scores[i])
#
#            # select words
#            new_words_lm_scores = np.array(replace_words_lm_scores[:-1])
#            # abs_diff_lm_scores = np.abs(new_words_lm_scores - replace_words_lm_scores[-1])
#            # rank_replaces_by_lm = np.argsort(abs_diff_lm_scores)
#            rank_replaces_by_lm = np.argsort(-new_words_lm_scores)
#
#            filtered_words_idx = rank_replaces_by_lm[self.top_n2:]
        if self.use_lm:
          prefix = ['']
          suffix = ['']
          if pos > 0 and pos<=self.n_prefix:
            prefix = [self.dataset.inv_dict[x_cur[pos-i-1]] for i in range(int(pos))[::-1]]
          elif pos>self.n_prefix:
            prefix = [self.dataset.inv_dict[x_cur[pos-i-1]] for i in range(self.n_prefix)[::-1]]
    
          
    #      orig_word = self.i_w_dict[seq[loc]]
          if self.use_suffix and pos < x_len-self.n_suffix and x_cur[pos+self.n_suffix]!=0:
            suffix = [self.dataset.inv_dict[x_cur[pos+i]] for i in range(1,self.n_suffix+1)]
          elif self.use_suffix and pos < x_len:
            suffix = [self.dataset.inv_dict[x_cur[pos+i]] for i in range(1,x_len-pos)]
          # print(self.dataset.inv_dict[x_orig[pos]])
          # print(prefix, suffix)
          word_list = [prefix+[self.dataset.inv_dict[w]]+suffix if w in self.dataset.inv_dict else prefix+['UNK']+suffix for w in replace_list[:self.top_n1]]
    # #      replace_words_orig = [self.dataset.inv_dict[w] if w in self.dataset.inv_dict else 'UNK' for w in replace_list[:self.top_n1]] + [orig_word]
         
          

          # print(seqs)
          # # library
          # seqs = [self.seq_list(seq) for seq in word_list]
          # replace_words_scores = self.scorer.sentence_score(seqs, reduce = 'prod')
          # new_words_scores = np.array(replace_words_scores)
          # rank_replaces_by_lm = np.argsort(new_words_scores)[::-1]

          replace_words_scores = self.lm.get_probs(word_list)
          rank_replaces_by_lm = np.argsort(replace_words_scores)

          # print(rank_replaces_by_lm)
          # print(np.array(seqs)[rank_replaces_by_lm])
          filtered_words_idx = rank_replaces_by_lm[self.top_n2:]
            # print(filtered_words_idx)
          new_x_scores[filtered_words_idx] = -10000000

        if (np.max(new_x_scores) > 0):
            return new_x_list[np.argsort(new_x_scores)[-1]]
        return x_cur
    def seq_list(self, prefix):
      sentence = ''

      for word in prefix:
          sentence += word + ' '
      sentence = sentence.strip()
      return sentence

    def perturb(self, x_cur, x_orig, neigbhours, neighbours_dist,  w_select_probs, target):
        # Pick a word that is not modified and is not UNK
        x_len = w_select_probs.shape[0]
        # to_modify = [idx  for idx in range(x_len) if (x_cur[idx] == x_orig[idx] and self.inv_dict[x_cur[idx]] != 'UNK' and
        #                                             self.dist_mat[x_cur[idx]][x_cur[idx]] != 100000) and
        #                     x_cur[idx] not in self.skip_list
        #            ]
        rand_idx = np.random.choice(x_len, 1, p=w_select_probs)[0]
        while x_cur[rand_idx] != x_orig[rand_idx] and np.sum(x_orig != x_cur) < np.sum(np.sign(w_select_probs)):
            # The conition above has a quick hack to prevent getting stuck in infinite loop while processing too short examples
            # and all words `excluding articles` have been already replaced and still no-successful attack found.
            # a more elegent way to handle this could be done in attack to abort early based on the status of all population members
            # or to improve select_best_replacement by making it schocastic.
            rand_idx = np.random.choice(x_len, 1, p=w_select_probs)[0]

        # src_word = x_cur[rand_idx]
        # replace_list,_ =  glove_utils.pick_most_similar_words(src_word, self.dist_mat, self.top_n, 0.5)
        replace_list = neigbhours[rand_idx]
        if len(replace_list) < self.top_n:
            replace_list = np.concatenate(
                (replace_list, np.zeros(self.top_n - replace_list.shape[0])))
        return self.select_best_replacement(rand_idx, x_len, x_cur, x_orig, target, replace_list)

    def generate_population(self, x_orig, neigbhours_list, neighbours_dist, w_select_probs, target, pop_size):
        return [self.perturb(x_orig, x_orig, neigbhours_list, neighbours_dist, w_select_probs, target) for _ in range(pop_size)]

    def crossover(self, x1, x2):
        x_new = x1.copy()
        for i in range(len(x1)):
            if np.random.uniform() < 0.5:
                x_new[i] = x2[i]
        return x_new

    def attack(self, x_orig, target, max_change=0.4):
        x_orig = x_orig.numpy().squeeze()
        x_adv = x_orig.copy()
        x_len = np.sum(np.sign(x_orig))
        # Neigbhours for every word.
        tmp = [glove_utils.pick_most_similar_words(
            self.compute_dist(x_orig[i]), 50, 0.5) for i in range(x_len)]
        neigbhours_list = [x[0] for x in tmp]
        neighbours_dist = [x[1] for x in tmp]
        neighbours_len = [len(x) for x in neigbhours_list]
        for i in range(x_len):
            if (x_adv[i] < 27):
                # To prevent replacement of words like 'the', 'a', 'of', etc.
                neighbours_len[i] = 0
        w_select_probs = neighbours_len / np.sum(neighbours_len)
        tmp = [glove_utils.pick_most_similar_words(
            self.compute_dist(x_orig[i]), self.top_n, 0.5) for i in range(x_len)]
        neigbhours_list = [x[0] for x in tmp]
        neighbours_dist = [x[1] for x in tmp]
        pop = self.generate_population(
            x_orig, neigbhours_list, neighbours_dist, w_select_probs, target, self.pop_size)
        for i in range(self.max_iters):
            # print(i)
            l_tensor = x_len*torch.ones([len(pop)])
            pop_np = np.expand_dims(pop[0], 0)
            for p in pop[1:]:
              pop_np = np.concatenate((pop_np, np.expand_dims(p, 0)),0) 
            
            pop_tensor = torch.tensor(pop_np).type(torch.LongTensor).to(self.device)
            l_tensor = l_tensor.to(self.device)
            self.batch_model.eval()
            with torch.no_grad():
              pop_preds = self.batch_model.pred(pop_tensor, l_tensor, False)[1].cpu().detach().numpy()

            
#            pop_preds = self.batch_model.predict(self.sess, np.array(pop))
            pop_scores = pop_preds[:, target]
            print('\t\t', i, ' -- ', np.max(pop_scores))
            pop_ranks = np.argsort(pop_scores)[::-1]
            top_attack = pop_ranks[0]
            ampl = pop_scores / self.temp
            # print(ampl)
            covariance = np.cov(ampl)
#            print(covariance)
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
            select_probs = logits / np.sum(logits)
            # print(select_probs)

            if np.argmax(pop_preds[top_attack, :]) == target:
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
                x, x_orig, neigbhours_list, neighbours_dist, w_select_probs, target) for x in childs]
            pop = elite + childs

        return None
