"""
Cecilia Aponte

Load Trained Model and Predict Tokens
"""

import numpy as np
from typing import List, Tuple
import os, pickle

from model import Model
import torch

import sys, math

#sys.path.append(os.path.join(os.getcwd(), "..", "hw1/stud")) # TESTING
sys.path.append(os.path.join(os.getcwd(), "hw1/stud"))

import trainer
from trainer import Vocab, NERTaggingData, HypParams, NER_Model
import __main__
__main__.Vocab = trainer.Vocab


def build_model(device: str) -> Model: 
    
    directory = os.getcwd()
    # For TESTING
#    dataset_dir = os.path.join(directory, '..', "data")
#    model_dir = os.path.join(directory, '..', "model")
    
    dataset_dir = os.path.join(directory, "data")
    model_dir = os.path.join(directory, "model")
    model_name = '05_capTot_embed_Bi-LSTM' #'06_partSenseTag_capTot_embed_Bi-LSTM'
    saved_model_path = os.path.join(model_dir, model_name + ".pt")
    
    model = StudentModel(model_dir, dataset_dir, device) 
    model.nertag_Model.load_state_dict(torch.load(saved_model_path, map_location = device))
    
    return model



class StudentModel(Model):
    
    def __init__(self, model_dir, dataset_dir, device):
        super(StudentModel, self).__init__()
        
        self.window_size, self.window_shift = 100, 100
        self.device = device
        
        # Load vocabularies
        with open(os.path.join(model_dir, "vocabulary.pkl"),"rb") as file:  
            self.vocabulary = pickle.load(file)
        
        with open(os.path.join(model_dir, "label_vocabulary.pkl"),"rb") as file:  
            self.label_vocabulary = pickle.load(file)
    
        # Load pre-trained embedings
        with open(os.path.join(model_dir, "pretrained_embed.pkl"),"rb") as file:  
            pretrained_embed = pickle.load(file)

        # Model parameters
        model_param = HypParams(self.vocabulary, self.label_vocabulary, 400, pretrained_embed)

        self.nertag_Model = NER_Model(model_param)
        
    

    def predict(self, tokens: List[List[str]]) -> List[List[str]]:

        self.nertag_Model.eval()
        
        # Pre-process tokens
        data = NERTaggingData('None', self.window_size, self.window_shift, pred=tokens, lowercase=False)
        data.encode_dataset(self.vocabulary, self.label_vocabulary)

                    
        predictions = list()
        with torch.no_grad():
            encoded_input = [x for x in data] 
                    
            index_encode = 0
            for i, raw_item in enumerate(tokens):
                
                test_item = encoded_input[index_encode]
                output = self.nertag_Model(test_item.unsqueeze(0))
                prediction = NERTaggingData.get_decode_index2text(output, self.label_vocabulary)[0]
                prediction = prediction[:len(raw_item)]
                
                # In the cases where the window size has been reached its maximum
                # so the sentence is encoded in more than one item
                if len(raw_item) != len(encoded_input[index_encode])- encoded_input[index_encode].tolist().count(0):
                    extra_items = len(raw_item) % self.window_size
                    index_encode += math.ceil(len(raw_item) / self.window_size)
                    test_item = encoded_input[index_encode]
                    output = self.nertag_Model(test_item.unsqueeze(0))
                    extra_decoded = NERTaggingData.get_decode_index2text(output, self.label_vocabulary)[0]
                    prediction += extra_decoded[:extra_items]

                else: 
                    index_encode += 1
                    
                predictions.append(prediction)
                
        return predictions
    



#def build_model(device: str) -> Model:
#    # STUDENT: return StudentModel()
#    # STUDENT: your model MUST be loaded on the device "device" indicates
#    return RandomBaseline()

#class RandomBaseline(Model):
#
#    options = [
#        ('LOC', 98412),
#        ('O', 2512990),
#        ('ORG', 71633),
#        ('PER', 115758)
#    ]
#
#    def __init__(self):
#
#        self._options = [option[0] for option in self.options]
#        self._weights = np.array([option[1] for option in self.options])
#        self._weights = self._weights / self._weights.sum()
#
#    def predict(self, tokens: List[List[str]]) -> List[List[str]]:
#        return [[str(np.random.choice(self._options, 1, p=self._weights)[0]) for _x in x] for x in tokens]




# TESTING ONLY
#quick testing:
#tokens = [['This', 'is', 'the', 'first', 'homework'],  ['Barack', 'Obama', 'was', 'elected']]
#labels = [['O', 'O', 'O', 'O', 'O'], ['PER', 'PER', 'O', 'O']]

# test with real data
#from conllu import parse 
#dataset_file = os.path.join(os.getcwd(), '..', "data/dev.tsv")
#with open(dataset_file) as reader:
#    sentences = parse(reader.read())
#tokens = list()
#for sentence in sentences:
#    sentlist= list()
#    for word_raw in list(sentence):
#            sentlist.append(word_raw['form'])
#    tokens.append(sentlist)
#model = build_model('cpu')
#model.predict(tokens)
