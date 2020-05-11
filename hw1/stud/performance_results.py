#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ccapontep
"""

# For plotting results:
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
import torch, os, pickle

# Get path for each file type
directory = os.getcwd()
model_dir = os.path.join(directory, "..", "..", "model")

with open(os.path.join(model_dir, "label_vocabulary.pkl"),"rb") as file:  
    label_vocabulary = pickle.load(file)

# Computes the performance of the model for different datasets needed
def compute_performance(model, #:torch.nn.Module, 
                        dataset, #:DataLoader, 
                        label_vocab): #:Vocab):
    predictions = list()
    labels = list()

    model.eval()

    with torch.no_grad():
        for item in dataset:
            inputs = item["inputs"]
            outputs = item["outputs"].tolist()

            prediction = model(inputs)
            prediction = torch.argmax(prediction, -1).tolist() 

            for i, item in enumerate(outputs):
                max_index = len(item) - item.count(0)
                labels += item[:max_index]
                predictions += prediction[i][:max_index]
                # print(prediction[i][:max_index])
                # print(item[:max_index], '\n')


    # macro precision, recall, f1: computes metric and then averages across class.  
    # Doesn't take into account the quantity of samples per class
    macro_precision = precision_score(labels, predictions, average="macro", zero_division=0)
    macro_recall = recall_score(labels, predictions, average="macro", zero_division=0)
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)

    # To check the precision per class
    per_class_f1 = f1_score(labels, predictions, labels = list(i for i, item in enumerate(label_vocabulary.str2int)), average=None, zero_division=0)

    # Confusion matrix
    confusionMatrix = confusion_matrix(labels, predictions, labels = [i for i in label_vocabulary.str2int.values()][1:], normalize='true')


    output_performance = {"macro_precision":macro_precision,
                          "macro_recall":macro_recall, 
                          "macro_f1":macro_f1,
                          "per_class_f1":per_class_f1,
                          "confusionMatrix":confusionMatrix}

    return output_performance

# Gets the performance for train, dev, and testing
# Then, plots a table with recall, precision and f1 (overall and per class) 
def performance_heatmap(nertag_Model, train_dataset, dev_dataset, test_dataset):
    
    performance_train = compute_performance(nertag_Model, train_dataset, label_vocabulary)
    performance_dev = compute_performance(nertag_Model, dev_dataset, label_vocabulary)
    performance_test = compute_performance(nertag_Model, test_dataset, label_vocabulary)
    
    # Overall Precision, Recall and F1
    print("Performance Type\tTrain\t\tValidation\tTest")
    print("="*65)
    print("Precision:\t\t{:0.3f}\t\t{:0.3f}\t\t{:0.3f}".format(performance_train["macro_precision"], performance_dev["macro_precision"], performance_test["macro_precision"]))
    print("Recall:\t\t\t{:0.3f}\t\t{:0.3f}\t\t{:0.3f}".format(performance_train["macro_recall"], performance_dev["macro_recall"], performance_test["macro_recall"]))
    print("F1:\t\t\t{:0.3f}\t\t{:0.3f}\t\t{:0.3f}".format(performance_train["macro_f1"], performance_dev["macro_f1"], performance_test["macro_f1"]))
    print("-"*65)
    print('F1 per Class')
    
    # Per Class F1
    train_perClass = list()
    dev_perClass = list()
    test_perClass = list()
    #train
    for id_class, performance in sorted(enumerate(performance_train["per_class_f1"]), key=lambda elem: -elem[1]):
        label = label_vocabulary.int2str[id_class]
        train_perClass += [label, performance]
    #dev
    for id_class, performance in sorted(enumerate(performance_dev["per_class_f1"]), key=lambda elem: -elem[1]):
        label = label_vocabulary.int2str[id_class]
        dev_perClass += [label, performance]
    #test
    for id_class, performance in sorted(enumerate(performance_test["per_class_f1"]), key=lambda elem: -elem[1]):
        label = label_vocabulary.int2str[id_class]
        test_perClass += [label, performance]
    
    # Place results of per Class in table
    labels = ['O', 'PER', 'LOC', 'ORG']
    for item in labels:
        print("\t{}:\t\t{:0.3f}\t\t{:0.3f}\t\t{:0.3f}".format(item, train_perClass[train_perClass.index(item)+1], dev_perClass[dev_perClass.index(item)+1], test_perClass[test_perClass.index(item)+1]))
    


    # Plots a heat map confusion matrix for each train, dev, test
    
    # Confusion Matrix for TRAINING
    df_confusionMatrix = pd.DataFrame(performance_train["confusionMatrix"], index = [i for i in label_vocabulary.int2str[1:]],
                      columns = [i for i in label_vocabulary.int2str[1:]])
    ax = plt.axes()
    sn.heatmap(df_confusionMatrix, annot=True, annot_kws={"size": 11}, cmap='Oranges', ax = ax, fmt='.2%')
    ax.set_title('NER Normalized Confusion Matrix for Training Data', pad=20)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('Ground Truth Labels')
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 25, fontsize = 10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 25, fontsize = 10)
    plt.savefig('images/CM_train.png', bbox_inches='tight', pad_inches=0.2)
    plt.show()
    
    # Confusion Matrix for VALIDATION
    df_confusionMatrix = pd.DataFrame(performance_dev["confusionMatrix"], index = [i for i in label_vocabulary.int2str[1:]],
                      columns = [i for i in label_vocabulary.int2str[1:]])
    ax = plt.axes()
    sn.heatmap(df_confusionMatrix, annot=True, annot_kws={"size": 11}, cmap='Blues', ax = ax, fmt='.2%')
    ax.set_title('NER Normalized Confusion Matrix for Validation Data', pad=20)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('Ground Truth Labels')
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 25, fontsize = 10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 25, fontsize = 10)
    plt.savefig('images/CM_dev.png', bbox_inches='tight', pad_inches=0.2)
    plt.show()
    
    # Confusion Matrix for TESTING
    df_confusionMatrix = pd.DataFrame(performance_test["confusionMatrix"], index = [i for i in label_vocabulary.int2str[1:]],
                      columns = [i for i in label_vocabulary.int2str[1:]])
    ax = plt.axes()
    sn.heatmap(df_confusionMatrix, annot=True, annot_kws={"size": 11}, cmap='Greens', ax = ax, fmt='.2%')
    ax.set_title('NER Normalized Confusion Matrix for Testing Data', pad=20)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('Ground Truth Labels')
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 25, fontsize = 10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 25, fontsize = 10)
    plt.savefig('images/CM_test.png', bbox_inches='tight', pad_inches=0.2)
    plt.show()
