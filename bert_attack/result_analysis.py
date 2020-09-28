# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 10:48:33 2020

@author: 13758
"""
import numpy as np
import matplotlib.pyplot as plt
# Amplification on 100L
sr_350 = np.array([92, 91, 86.8, 76.28])
sr_300 = np.array([92.89, 93.0,88.6, 77.8])
sr_250 = np.array([94, 95.6,89.8,78.6])
sr_200 = np.array([93.4,95.4,89.0,77.6])
len = np.array([200,150,100,50])
plt.figure()
plt.plot(len, sr_350, label='Sentence length of train: 350', marker='o')
plt.plot(len, sr_300, label='Sentence length of train: 300', marker='o')
plt.plot(len, sr_250, label='Sentence length of train: 250', marker='o')
plt.plot(len, sr_200, label='Sentence length of train: 200', marker='o')
# plt.plot(-np.log(temp), sr_25, label='Success rate (25% allowance)', marker='o')
# plt.plot(-np.log(0.0005), 99.6, label='Success rate&BN (25% allowance)', marker='*', markersize='10', color='r')
# plt.plot(-np.log(0.0005), 98.8, label='Success rate&BN (20% allowance)', marker='*', markersize='10', color='k')

# plt.ylim(ymin=95)
plt.title('Success rate without amplification of probability (LM: No)')
plt.xlabel('Length of sentences: attack')
plt.ylabel('Success rate (Percentage)')
plt.legend()
plt.show()


# Amplification on 100L
sr_20 = np.array([96.2, 96.2, 96.6, 96.2, 97, 98])
sr_25 = np.array([98.6, 98.6, 98.6, 99, 99, 99.2])
temp = np.array([1, 0.5, 0.2, 0.1, 0.05, 0.01])
plt.figure()
plt.plot(-np.log(temp), sr_20, label='Success rate (20% allowance)', marker='o', color='g')
plt.plot(-np.log(temp), sr_25, label='Success rate (25% allowance)', marker='o')
plt.scatter(-np.log(0.0005), 99.6, label='Success rate&BN (25% allowance)', marker='*', linewidths='5', color='k')
plt.scatter(-np.log(0.0005), 98.8, label='Success rate&BN (20% allowance)', marker='*', linewidths='5', color='r')

plt.ylim(ymin=95.5)
plt.title('Success rate with amplification of probability (Sentence length: 100; LM: No)')
plt.xlabel('Logrithm of amplification value')
plt.ylabel('Success rate (Percentage)')
plt.legend()
plt.show()

# Probability amplification with BN and wo BN on 90L
sr_bn = np.array([89.6, 90.2, 90.2, 92.2, 93, 93.04, 93.6])
temp_bn = np.array([0.5, 0.2, 0.1, 0.05, 0.01, 0.001, 0.0005])
sr = np.array([88.00, 87.2, 89.8, 88.8, 92.19])
temp = np.array([0.5, 0.2, 0.1, 0.05, 0.01])
plt.figure()
plt.plot(-np.log(temp_bn), sr_bn, label='Success rate with BN', marker='o')
plt.plot(-np.log(temp), sr, label='Success rate without BN', marker='*')
plt.ylim(ymin=86)
plt.title('Success rate with amplification of probability (Sentence length: 90; LM: No)')
plt.xlabel('Logrithm of amplification value')
plt.ylabel('Success rate (Percentage)')
plt.legend()
plt.show()

temp = np.array([1, 0.05, 0.01, 0.0005, 0.0003])
sr_20 = np.array([0.9110, 0.9090, 0.9260, 0.9320, 0.9450])*100
sr_25 = np.array([0.9550, 0.9550, 0.9620, 0.9680, 0.9730])*100
plt.figure()
plt.plot(-np.log(temp), sr_20, label='Success rate (20% allowance)', marker='o', color='g')
plt.plot(-np.log(temp), sr_25, label='Success rate (25% allowance)', marker='o')

plt.ylim(ymin=85)
plt.title('Success rate with amplification of probability (Sentence length: 100; LM: Y)')
plt.xlabel('Logrithm of amplification value')
plt.ylabel('Success rate (Percentage)')
plt.legend()
plt.show()
