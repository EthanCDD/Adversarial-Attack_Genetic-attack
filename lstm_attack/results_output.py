# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 08:21:53 2020

@author: 13758
"""
import os
import numpy as np

class ResultCalculation(object):
    def __init__(self, seq_s_path, seq_o_path):
        seq_s = np.load(os.path.join('results',seq_s_path), allow_pickle = True)
        seq_o = np.load(os.path.join('results',seq_o_path), allow_pickle = True)
        self.seq_s = seq_s.tolist()[:200]
        self.seq_o = seq_o.tolist()[:200]
        self.process()
    def process(self):
        orig_len = []
        normalised_dist = []
        for i in range(len(self.seq_s)):
            if None not in np.array(self.seq_s[i]):
                orig_len.append(np.sum(np.sign(self.seq_s[i])))
                normalised_dist.append(np.sum(self.seq_s[i] != self.seq_o[i])/np.sum(np.sign(self.seq_s[i])))                 
        SUCCESS_THRESHOLD  = 0.25
        successful_attacks = [x < SUCCESS_THRESHOLD for x in normalised_dist]
        print('Attack success rate : {:.2f}%'.format(np.sum(successful_attacks)/len(self.seq_o)*100))
        print('Median percentange of modifications: {:.02f}% '.format(
            np.median([x for x in normalised_dist if x < 1])*100))
        print('Mean percentange of modifications: {:.02f}% '.format(
            np.mean([x for x in normalised_dist if x < 1])*100))