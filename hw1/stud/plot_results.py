#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ccapontep

Plot Results for NER model
"""

import matplotlib.pyplot as plt
import numpy as np

# F1 Performance with changing Hidden Layer Size
f1_hidden = [0.795, 0.802, 0.801, 0.803, 0.804]
hidden = [32, 64, 128, 256, 400]

fig, ax = plt.subplots()
plt.plot(hidden, f1_hidden, color='b')
plt.xlabel('LSTM hidden layer size')
plt.ylabel('F1 Score')
plt.title('F1 Performance with changing Hidden Layer Size', pad=20)
ax.set_xticks(ticks=np.add(hidden,.2/2))
ax.set_xticklabels(hidden)

plt.savefig('images/f1_hidden.png', bbox_inches='tight', pad_inches=0.2)
plt.show()

# F1 Performance with changing Dropout
f1_drop = [0.786, 0.804, 0.802, 0.748]
dropout = [0.2, 0.4, 0.6, 0.8]

fig, ax = plt.subplots()
plt.plot(dropout, f1_drop, color='b')
plt.xlabel('LSTM Dropout')
plt.ylabel('F1 Score')
plt.title('F1 Performance with changing Dropout', pad=20)
ax.set_xticks(ticks=np.add(dropout,0.001))
ax.set_xticklabels(dropout)

plt.savefig('images/f1_drop.png', bbox_inches='tight', pad_inches=0.2)
plt.show()


# F1 Performance with changing Epochs
f1_epo = [0.723, 0.804, 0.790, 0.777]
epochs = [10, 20, 50, 100]

fig, ax = plt.subplots()
plt.plot(epochs, f1_epo, color='b')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.title('F1 Performance with changing Epochs', pad=20)
ax.set_xticks(ticks=np.add(epochs,0.001))
ax.set_xticklabels(epochs)

plt.savefig('images/f1_epochs.png', bbox_inches='tight', pad_inches=0.2)
plt.show()


# F1 Performance with changing LSTM Layer
f1_layer = [0.804, 0.801, 0.797]
LSTMlayer = [1, 2, 3]

fig, ax = plt.subplots()
plt.plot(LSTMlayer, f1_layer, color='b')
plt.xlabel('Number of LSTM Layers')
plt.ylabel('F1 Score')
plt.title('F1 Performance with changing number of LSTM Layers', pad=20)
ax.set_xticks(ticks=np.add(LSTMlayer,0.001))
ax.set_xticklabels(LSTMlayer)

plt.savefig('images/f1_lstmLayer.png', bbox_inches='tight', pad_inches=0.2)
plt.show()


# F1 Performance with changing Vocab Freq
f1_vocFreq = [0.802, 0.785, 0.772, 0.754]
VocabFreq = [1, 2, 3, 5]

fig, ax = plt.subplots()
plt.plot(VocabFreq, f1_vocFreq, color='b')
plt.xlabel('Vocabulary Minimum Frequency')
plt.ylabel('F1 Score')
plt.title('F1 Performance with changing minimum frequency for Vocabulary', pad=20)
ax.set_xticks(ticks=np.add(VocabFreq,0.001))
ax.set_xticklabels(VocabFreq)

plt.savefig('images/f1_vocabFreq.png', bbox_inches='tight', pad_inches=0.2)
plt.show()


# F1 Performance with changing Window Size/Shift
f1_windSize = [0.783, 0.801, 0.803, 0.806, 0.802]
f1_windShift = [0.790, 0.793, 0.802, 0.802]

window_size = [10, 30, 60, 100, 130]
window_shift = [5, 20, 40, 70]
allTicks = [5, 10, 20, 30, 40, 60, 70, 100, 130]

fig, ax = plt.subplots()
plt.plot(window_size, f1_windSize, color='b', label='Window size/shift equal')
plt.plot(window_shift, f1_windShift, color='orange', label='Window shift change')

plt.xlabel('Window Size / Shift')
plt.ylabel('F1 Score')
plt.title('F1 Performance with changing Window Size /  Shift', pad=20)
ax.set_xticks(ticks=np.add(allTicks,0.001))
ax.set_xticklabels(allTicks)
plt.legend(loc="lower right")

plt.savefig('images/f1_window.png', bbox_inches='tight', pad_inches=0.2)
plt.show()


# F1 Performance with changing Optimizer
f1_opt = [0.796, 0.795, 0.684]
opt = ['Adam', 'Adagrad', 'Adadelta']

fig, ax = plt.subplots()
plt.plot(opt, f1_opt, color='b')
plt.xlabel('Optimizer Type')
plt.ylabel('F1 Score')
plt.title('F1 Performance with changing Optimizer Type', pad=20)
ax.set_xticklabels(opt)

plt.savefig('images/f1_optim.png', bbox_inches='tight', pad_inches=0.2)
plt.show()


# F1 Performance with changing model type part 1
f1_model1 = [0.6836, 0.686, 0.715, 0.801, 0.806]
#f1_model1 = [0.2513, 0.6936, 0.696, 0.715, 0.801, 0.805]

model1 = ['Basic LSTM', 'Basic Bi-LSTM', 'SensEmbed Bi-LSTM', 
          'SensEmbed Part-Cap Bi-LSTM', 'SensEmbed All-Cap Bi-LSTM' ]
#model1 = ['Random', 'Basic LSTM', 'Basic Bi-LSTM', 'POSEmbed Bi-LSTM', 'POSEmbed Partial-Capital Bi-LSTM', 'POSEmbed All-Capital Bi-LSTM' ]
# results from the first two are from test dataset, and from then on are from
# validation

fig, ax = plt.subplots()
plt.plot(model1, f1_model1, color='b')
plt.xlabel('Model Type')
plt.ylabel('F1 Score')
plt.title('F1 Performance with changing Model Type - Basics', pad=20)
ax.set_xticklabels(model1, rotation = 10, fontsize = 6)


plt.savefig('images/f1_model_1.png', bbox_inches='tight', pad_inches=0.2, dpi=200)
plt.show()

# F1 Performance with changing model type part 2
f1_model2 = [0.799, 0.807, 0.813, 0.813]
model2 = ['All-SenseTag', 
          'Part-SenseTag-common', 
          'Part-SenseTag-WSDsyn', 
          'Part-SenseTag-WSDall']

f1_model3 = [0.738, 0.731, 0.720, 0.724]

fig, ax = plt.subplots()
plt.plot(model2, f1_model2, color='b', label='Unbalanced')
plt.plot(model2, f1_model3, color='orange', label='Weighted Loss')

plt.xlabel('Model Type feature for SensEmbed Bi-LSTM')
plt.ylabel('F1 Score')
plt.title('F1 Performance with changing Model Type - Unbalanced/Weighted', pad=20)
ax.set_xticklabels(model2, rotation = 10, fontsize = 6)
plt.legend(loc="middle right")


plt.savefig('images/f1_model_2.png', bbox_inches='tight', pad_inches=0.2, dpi=200)
plt.show()


