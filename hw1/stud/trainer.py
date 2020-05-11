# -*- coding: utf-8 -*-

"""
Cecilia Aponte

Trainer File - Original is trained in Colab, see trainer.ipynb file
"""

import os
import itertools
import pickle
from conllu import parse 
import torch
import numpy as np
from tqdm import tqdm
from collections import Counter
from torch.utils.data import DataLoader
from collections import defaultdict 
from gensim.models import KeyedVectors


# For POS tag pre-processing Sense Tagging (Optional Step)
#import nltk
#nltk.set_proxy('http://127.0.0.1:12345')
#nltk.download('averaged_perceptron_tagger')
#nltk.download("wordnet")
#from nltk.stem.wordnet import WordNetLemmatizer
#from nltk.corpus import wordnet as wn
#import csv
#from nltk.wsd import lesk


# Get path for each file type
directory = os.getcwd()
dataset_dir = os.path.join(directory, "..", "..", "data")
model_dir = os.path.join(directory, "..", "..", "model")
stud_dir = os.path.join(directory, "hw1/stud")


file_train = os.path.join(dataset_dir, "train.tsv")
file_dev = os.path.join(dataset_dir, "dev.tsv")
file_test = os.path.join(dataset_dir, "test.tsv")
embed_file = os.path.join(model_dir, "embeddings.vec")


"""# Data Preparation

The following are done in this section:

*   Create Vocab Class to create the vocabularies
*   Create Sense tokenizer functions for the dataset
*   Create the NERTaggingData class to create an encoded Dataset
*   Get the NER tagged datasets
*   Build vocabulary for text and labels --> corresponding indices
*   Encode the dataset each sample with its corresponding indices given by the vocabs
*   Update the datasets to include the batch size 
"""

######################### VOCABULARY CREATION CLASS ##########################
class Vocab:
    def __init__(self, counter, special_tokens, min_freq=1):

        # Initiate dictionary and reverse dictionary
        self.str2int = defaultdict()
        self.int2str = list()
        self.special_tokens = special_tokens
        
        # Add special tokens first to the dictionary
        if len(special_tokens) > 0:
            for i, word in enumerate(special_tokens):
                self.str2int[word] = i
                self.int2str.append(word)
                
                # Set the variable unk_index
                if word == '<unk>':
                    self.unk_index = i
            
        
        # Add tokens form/label to the dictionaries that have a minimum frequency
        counting = 0 + len(special_tokens)
        for i, word in enumerate(counter):
            if counter[word] >= min_freq:
                self.str2int[word] = counting
                self.int2str.append(word)
                counting += 1

      
        
    def __len__(self):
          return len(self.str2int)

    # Used to get an item from the dictionaries 
    def __getitem__(self, key):
        if isinstance(key, int):
            try:
                item = self.int2str[key]
            except:
                # If out of range, will suggest another value within and return None for the item
                item = print('Key index is out of range. Please choose an integer value between 0 and', len(self.str2int))

            
        if isinstance(key, str):
            try:
                item = self.str2int[key]
            except:
                # If string key is not in the dictionary, will return index for UNK - unknown
                item = self.str2int['<unk>']            
            
            
        return item

    # Similar to the variable used in TorchText Vocab, does it above much simpler without needing to call this variable
    def str2int(self):
        def __getitem__(self, key):
            try:
                item = self.str2int[key]
            except:
                item = self.str2int['<unk>']         
        return item

    # Similar to the variable used in TorchText Vocab, does it above much simpler without needing to call this variable
    def int2str(self):
        def __getitem__(self, key):
            try:
                item = self.int2str[key]
            except:
                item = print('Key index is out of range. Please choose an integer value between 0 and', len(self.str2int))
        return item
    
    # Load the pre-trained vectors into the dictionary. The previous vocab and
    # these vectors will have to be aligned to update the integers -> vectors,
    # add any missing words, and make 'unk' those words that aren't in the vectors
    def load_vectors(self, w2v_model):
        # Add special tokens first to the dictionary
        if len(self.special_tokens) > 0:
            if '<unk>' in self.special_tokens: 
                w2v_model.add('<unk>', np.positive(w2v_model.get_vector('a')*0), replace=False)

            for i, word in enumerate(self.special_tokens):
                # add special tokens to the word2vec model
                change = 0.00001 * i
                if word != '<unk>': w2v_model.add(word, np.positive(w2v_model.get_vector('<unk>'))+change,replace=False)
        
        vocab_embed = list((w2v_model.vocab).keys())
        
        return vocab_embed, w2v_model
       
######################      SENSE TOKENIZER      ############################## 
    
# Optional - Not implemented in final model
# Following functions are to add babelnet id for each word that has a Sense tag
# The word's form is updated with this. This is used for the word sense-embeddings
# since these have words as: word_bnID

# Get wordnet POS only available Verb, Noun, Adverb, Adjective
def get_wordnet_pos(word_tag):
    if word_tag.startswith('J'): tag= wn.ADJ
    elif word_tag.startswith('V'): tag= wn.VERB
    elif word_tag.startswith('N'): tag= wn.NOUN
    elif word_tag.startswith('R'): tag= wn.ADV
    else: tag= None 

    return tag

# Get 1) lemma from word and POS tag 2) synset using lemma and POS
# 3) wordnet number with sysnet offset and pos 4) babelnet number from 
# bn2wn dictionary created from file. Create new form by combining word_bn#    
def get_newForm(sentList, dic_wn2bn, word, wnPOS, word_id):
    lemmatizer = WordNetLemmatizer()
    
    if wnPOS != None: #and (word[0].islower() and word_id != 0): 
        try:
            lemma = lemmatizer.lemmatize(word, pos=wnPOS) 
            syn = wn.synsets(lemma, pos=wnPOS)[0] # get most common synset        
#            wsd_syn = lesk(sentList, lemma, pos=wnPOS) #, synsets=syn)
            wnNum = 'wn:' + str(syn.offset()).zfill(8) + syn.pos()
            try: 
                bnNum = dic_wn2bn[wnNum]
                new_form = lemma + "_" + bnNum
            except: new_form = 'None'
        except: new_form = 'None'
    else: new_form = 'None'
    
    return new_form

# tokenize with Sense tag
def Sense_tokenize(sentences, pred=None):
    # Get file wordnet to babelnet dictionary from file
    dic_wn2bn = dict()
    with open(os.path.join(stud_dir, 'babelnet2wordnet.tsv'), 'r')  as fileBn2Wn:
        file_bn2wn = csv.reader(fileBn2Wn, delimiter="\t", quotechar='"')
        for row in file_bn2wn:
              dic_wn2bn[row[1]] = row[0]

    list_newForm = list()

    # Create a list of all the new forms to be later updated in dataset
    for i_sent, sentence in enumerate(sentences):
        sentList = list()
        if pred == None:
            for word_id, word_raw in enumerate(list(sentence)):
                    sentList.append(word_raw['form'])
        else: sentList = sentence

        # Get the babelnetID for each word that has a synset
        tagged = nltk.pos_tag(sentList)
        for word, tag in tagged:
            if pred != None:
                word_id = sentence.index(word)
            wnPOS = get_wordnet_pos(tag)
            new_form = get_newForm(sentList, dic_wn2bn, word, wnPOS, word_id)    
            list_newForm.append([i_sent, word_id, new_form])
            if new_form != 'None' and pred != None:
                sentences[i_sent][word_id] = new_form

    # Update each form with new form if available
    for i, item in enumerate(itertools.chain.from_iterable(sentences)):
        if list_newForm[i] != 'None':
            if pred == None:
                item['form'] = list_newForm[i]
        else: continue  
    

    del list_newForm, dic_wn2bn

    return sentences


#######################     NER TAGGER CLASS    ###############################

class NERTaggingData:

    def __init__(self, 
                 data_file:str, 
                 window_size:int, 
                 window_shift:int=-1,
                 lowercase=True, 
                 device="cpu",
                 pred=None):
                #  device="cuda"): # For colab
        """
        Args:
            data_file: The path to the dataset already tokenized to be loaded.
            
            window_size:  The maximum number of tokens in a sentence.
            
            window_shift: The number of tokens to shift the window in the
                          sentence. Default value is -1: window will be 
                          shifted by window_size.
            
            lowercase:  Whether the text has to be lowercased or not. For embeddings
                        with uppercase words, uppercase is useful (put as False)
            
            device:  device to run tensors (cpu or cuda).
            
        Output:
            encoded_data: List of samples with each token and its NER tag
        """

        self.data_file = data_file
        self.window_size = window_size
        self.window_shift = window_shift if window_shift > 0 else window_size
        self.lowercase = lowercase
        self.pred = pred


        # read and parse entire data file
        if self.pred == None:
            with open(data_file) as reader:
                sentences = parse(reader.read())
                
            # lowers each token, if envoked
            if self.lowercase:
                for item in itertools.chain.from_iterable(sentences):
                    item["form"] = item["form"].lower()
        else: 
            sentences = self.pred
            if self.lowercase:
                sentences = eval(str(sentences).lower())

        # OPTIONAL
        # add babelnet id to each word, if available, to add meaning given their Sense tag
#        sentences = Sense_tokenize(sentences, self.pred)

        self.device = device
        self.data = self.divide_windows(sentences)
        self.encoded_data = None

    # Get the length of the data
    def __len__(self):
        return len(self.data)

    # Returns a sample from the encoded dataset given an index
    def __getitem__(self, idx):
        if self.encoded_data is None:
            raise RuntimeError("""Have to call 'encode_dataset' on this object
            before trying to retrieve elements. To retrieve the original elements, 
            use the method get_windowItem(idx)""")
        return self.encoded_data[idx]


    # Returns a item widonw in the original data structure (without encoding)
    def get_windowItem(self, idx):
        return self.data[idx]


    def divide_windows(self, sentences):
        """ 
        Args:
            sentences:  list of lists of dictionaries (each represents a parsed word occurrence)

        Output:
            data: data that is divided in equal amounts of tokens for each sentence.
                  based on windows size and shift. If extra tokens, an empty None
                  pad is added.
        """
        data = []
        for sentence in sentences:
            for i in range(0, len(sentence), self.window_shift):
                win_size = len(sentence)-i
                if win_size >= self.window_size: window = sentence[i:i+self.window_size]
                else: window = sentence[i:i+self.window_size] + [None]*(self.window_size - win_size)
                data.append(window)
        return data

 
    def encode_dataset(self, vocabulary, label_vocabulary):
        """ 
        Args:
            vocabulary: vocabulary with mappings from words to indices and viceversa

            label_vocabulary: vocabulary with mappings from a string label 
                                to its corresponding index and vice versa

        Output:
            encoded_data: data encoded into inputs and outputs for each window.
                          For windows with None items, a pad is added to its label.
        """
        self.encoded_data = list()
        for i in range(len(self.data)):
            window = self.data[i]
            encoded_text = torch.LongTensor(self.get_encode_text2index(window, vocabulary)).to(self.device)
            
            if self.pred == None:
                encoded_labels = torch.LongTensor(self.get_encode_label(window, label_vocabulary)).to(self.device)
    
                self.encoded_data.append({"inputs":encoded_text, 
                                          "outputs":encoded_labels})
    
            else: self.encoded_data.append(encoded_text)


    @staticmethod
    # Get the encoded label for each item in the from of a dictionary in the window
    # If the item is None, the label becomes <pad>
    def get_encode_label(window:list, 
                         label_vocabulary:Vocab):
        label_out = list()
        for item_dic in window:
            if item_dic is not None: 
                label_out.append(label_vocabulary[item_dic["lemma"]])
            else: 
                label_out.append(label_vocabulary["<pad>"])
        return label_out

    # Get the encoded index of the text for each item in the from of a dictionary in the window
    # If the item is None, the label becomes <pad>
    def get_encode_text2index(self, window, vocabulary):

        indices = list()
        for item_dic in window:

            if self.pred == None and item_dic is not None: 
                token = item_dic["form"]
            else: token = item_dic
        
            if item_dic is None: 
                indices.append(vocabulary["<pad>"])
            # if word exists in vocabulary after converting the word string to integer
            elif token in vocabulary.str2int: 
                if self.pred == None: indices.append(vocabulary[item_dic["form"]])
                else: indices.append(vocabulary[item_dic])
            # if word doesn't exist in vocabulary, add UNK indices for unknown
            else: indices.append(vocabulary.unk_index) 
                
        return indices


    @staticmethod
    # Get a list of the decoded text from the output of NN model in the from of a tensor
    def get_decode_index2text(model_outputs, label_vocabulary):
        """
        Args:
            model_output:   a Tensor with shape (batch_size, max_len, label_vocab_size)
                            containing the output of the neural network model.

            l_label_vocabulary: vocabulary with mappings from a string label 
                                to its corresponding index and vice versa

        Output:
            The method returns a list of batch_size length where each element is a list
            of labels, one for each input token.
        """        
        max_indices = torch.argmax(model_outputs, -1).tolist() 
        predictions = list()

        # get word text in vocabulary after converting the index integer to word string
        for indices in max_indices:
            predictions.append([label_vocabulary.int2str[i] for i in indices])
        return predictions



# Build a vocabulary for each word and label found as a token dictionary
def build_vocab(dataset, w2v_model, min_freq=1):
    counter_vocab = Counter()
    counter_label = Counter()

    for i in tqdm(range(len(dataset))):
        for token in dataset.get_windowItem(i):
            if token is not None:
                counter_vocab[token["form"]] += 1
                counter_label[token["lemma"]] += 1
    
    # Special tokens added for padding and unknown words at testing time
    vocabulary = Vocab(counter_vocab, special_tokens=['<pad>', '<unk>'], min_freq=min_freq)
    print('Vocabulary length is:  ', len(vocabulary))
    
    if w2v_model != None:
        vocab_embed, w2v_model = vocabulary.load_vectors(w2v_model)
        print('Updated embedding with the special tokens.')

    label_vocabulary = Vocab(counter_label, special_tokens=['<pad>'])

    return vocabulary, label_vocabulary, vocab_embed, w2v_model



"""# Model Building

The following is included in this section:

*   Define the Hyperparameter Class
*   Create the NER Model Class
*   Create the Trainer Class
"""

class HypParams():
    def __init__(self, vocabulary, label_vocabulary, embed_dim, embeddings=None):
        self.vocabulary = vocabulary
        self.label_vocabulary = label_vocabulary
        self.vocab_size = len(self.vocabulary)
        self.hidden_dim = 64
        self.embedding_dim = embed_dim
        self.num_classes = len(self.label_vocabulary) 
        self.bidirectional = True
        self.num_layers = 2
        self.dropout = 0.5
        self.embeddings = embeddings


class NER_Model(torch.nn.Module):

    def __init__(self, model_param):
        super(NER_Model, self).__init__()

        # Embedding layer: a matrix (shape: vocab_size, embedding_dim). Where 
        # each index represents a word
        self.word_embedding = torch.nn.Embedding(model_param.vocab_size, 
                                                 model_param.embedding_dim) #,

        if model_param.embeddings is not None:
            print("Initializing embeddings layer from Pre-trained Sense Embeddings..")
            self.word_embedding.weight.data.copy_(model_param.embeddings)

        # Bi-LSTM 
        self.lstm = torch.nn.LSTM(input_size = model_param.embedding_dim, 
                                  num_layers = model_param.num_layers,
                                  hidden_size = model_param.hidden_dim,                                    
                                  dropout = model_param.dropout,
                                  bidirectional = model_param.bidirectional)
        
        lstm_output_dim = model_param.hidden_dim * 2

        self.dropout = torch.nn.Dropout(model_param.dropout)
        # fully connected layer
        self.classifier = torch.nn.Linear(lstm_output_dim, model_param.num_classes)

    def forward(self, x):
        embeddings = self.word_embedding(x)
        embeddings = self.dropout(embeddings)
        o, (h, c) = self.lstm(embeddings)
        o = self.dropout(o)
        output = self.classifier(o)
        
        return output



# Class to train and evaluate a model
class Trainer:
  
    def __init__(self, model, loss_function, optimizer, label_vocab):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.label_vocab = label_vocab

    def train(self, train_dataset, valid_dataset, epochs:int=2):

        print('Training ...')
        train_loss = 0.0
        for epoch in range(epochs):
            print(' Epoch {:03d}'.format(epoch + 1))

            epoch_loss = 0.0
            self.model.train()

            for step, item in enumerate(train_dataset):
                inputs = item['inputs']
                labels = item['outputs']
                self.optimizer.zero_grad()

                #input data to model to train and output the prediction
                pred = self.model(inputs)
                pred = pred.view(-1, pred.shape[-1])
                labels = labels.view(-1)
                
                loss = self.loss_function(pred, labels)
                loss.backward(create_graph=True)
                self.optimizer.step()

                epoch_loss += loss.tolist()
                
                if step % 500 == 0:
                    print('\tStep {}: current loss = {:0.4f}'.format(step, epoch_loss / (step + 1)))
            
            avg_epoch_loss = epoch_loss / len(train_dataset)
            train_loss += avg_epoch_loss
            
            print('\tAverage Train loss = {:0.4f}'.format(avg_epoch_loss))

        print('Done Training!')
        avg_epoch_loss = train_loss / epochs
        
        return avg_epoch_loss



window_size, window_shift = 100, 100

def main():

    # Get the NER tagged datasets
    train_dataset = NERTaggingData(file_train, window_size, window_shift, lowercase=False)
    dev_dataset = NERTaggingData(file_dev, window_size, window_shift, lowercase=False)
    # test_dataset = NERTaggingData(file_test, window_size, window_shift)

    
    nameFiles = ["vocabulary.pkl", "label_vocabulary.pkl", "pretrained_embed.pkl"]
    savedFiles = [idx for idx in os.listdir(model_dir) if idx.endswith('.pkl')]
    
    if all(elem in savedFiles  for elem in nameFiles):
        
        embed_dim = 400
    
        # Load vocabulary
        with open(os.path.join(model_dir, "vocabulary.pkl"),"rb") as file:  
            vocabulary = pickle.load(file)
    
        with open(os.path.join(model_dir, "label_vocabulary.pkl"),"rb") as file:  
            label_vocabulary = pickle.load(file)
    
        # Get pre-trained embedings
        with open(os.path.join(model_dir, "pretrained_embed.pkl"),"rb") as file:  
            pretrained_embed = pickle.load(file)
    
    else:
    
        # Load the word2vec model that has been pre-trained, and get its vocabulary
        # Get the weights of the embeddings to use later in the NER tag model
        w2v_model = KeyedVectors.load_word2vec_format(embed_file, binary = False)
        vocab_embed = list((w2v_model.vocab).keys())
        embed_dim = w2v_model.vector_size

        # Build vocabulary for text and labels --> corresponding indices
        vocabulary, label_vocabulary, vocab_embed, w2v_model = build_vocab(train_dataset, w2v_model, min_freq=1)
    
        # Get pre-trained embedings
        pretrained_embed = torch.randn(len(vocabulary), embed_dim)
        initialized = 0
        for i, word in enumerate(vocabulary.int2str):
            if word in vocab_embed:
                initialized += 1
                pretrained_embed[i] = torch.from_numpy(w2v_model[word].copy())
        print('Amount of initialized pre-trained embedings are: ', initialized)
    
        # Save the vocabularies
        with open(os.path.join(model_dir,"vocabulary.pkl"),"wb+") as file:  
            pickle.dump(vocabulary, file)
    
        with open(os.path.join(model_dir,"label_vocabulary.pkl"),"wb+") as file:  
            pickle.dump(label_vocabulary, file)
    
        with open(os.path.join(model_dir,"pretrained_embed.pkl"),"wb+") as file:  
            pickle.dump(pretrained_embed, file)
    
        print('Items saved..')
            
        del w2v_model, vocab_embed


       
    # Encode the dataset each sample with its corresponding indices given by the vocabs
    train_dataset.encode_dataset(vocabulary, label_vocabulary)
    dev_dataset.encode_dataset(vocabulary, label_vocabulary)
    # test_dataset.encode_dataset(vocabulary, label_vocabulary)
    
    # Update the datasets to include the batch size 
    train_dataset = DataLoader(train_dataset, batch_size=128)
    dev_dataset = DataLoader(dev_dataset, batch_size=128)
    # test_dataset = DataLoader(test_dataset, batch_size=128)        
        
    model_param = HypParams(vocabulary, label_vocabulary, embed_dim, pretrained_embed)

    nertag_Model = NER_Model(model_param) #.cuda()

    
    # weights to balance dataset, since "O" has too many samples and its mainly being chosen
    ratio = [0, 2177423/100409, 2177423/2177423, 2177423/61988, 2177423/84937]
    weight_loss = torch.FloatTensor(ratio)
    
    trainer = Trainer(
        model = nertag_Model,
        loss_function = torch.nn.CrossEntropyLoss(ignore_index=label_vocabulary['<pad>']), #, weight=weight_loss),
        optimizer = torch.optim.Adam(nertag_Model.parameters()),
        label_vocab=label_vocabulary)
    
    trainer.train(train_dataset, dev_dataset, 1)
    
    # Save model weights
    model_name = 'test'
    saved_model_path = os.path.join(model_dir, model_name + ".pt")
    output = open(saved_model_path, mode="wb+")
    torch.save(nertag_Model.state_dict(), output)
    output.close()


if __name__ == "__main__":
   # stuff only to run when not called via 'import'
   main()
   
   
# Output from testing in cpu and test.bash
"""
Initializing embeddings layer from Pre-trained Sense Embeddings..
Training ...
 Epoch 001
	Step 1: current avg loss = 0.5771
	Step 500: current avg loss = 0.4424
	Step 1000: current avg loss = 0.3882
	Step 1500:  current avg loss = 0.3581
	Step 2000:  current avg loss = 0.3390
	Step 2500:  current avg loss = 0.3257
	Step 3000:  current avg loss = 0.3161
	Average Train loss = train loss = 0.3091
Done Training!
"""