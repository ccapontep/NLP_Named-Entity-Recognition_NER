"""
Author: Cecilia Aponte
Natural Language Processing

Word2Vec SenseEmbed
"""

# Import libraries
import os, time
from SensEmbed_preprocess import getData_Embed_Train
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import multiprocessing, itertools
import pandas as pd
import pickle

# Set working directory
directory = os.getcwd()

# Set directory with dataset
dataset_dir = os.path.join(directory, "..", "resources")
embed_file = os.path.join(dataset_dir, "embeddings.vec")
wordsim_file = os.path.join(dataset_dir, "wordsim353/combined.tab")
EvalOut_SensEmbed_file = os.path.join(dataset_dir, "SenseEmbed_Output_Processed/EvalOut_SensEmbed.csv")
corpus_file = os.path.join(dataset_dir, "SenseEmbed_Output_Processed/w2vCorpus")

class IterableCorpus(object):
   def __init__(self, dir_name):
      self.dir_name = dir_name
   def __iter__(self):
      for idx,file_name in enumerate(os.listdir(self.dir_name)):
#          print('Reading corpus from file: ', file_name)
        for idxx,line in enumerate(pickle.load(open(os.path.join(self.dir_name, file_name),'rb')).split('\n')):
            words = [word for word in line.split(' ')]
            yield words

class IterLogger(CallbackAny2Vec):
    '''Callback to log information about training'''

    def __init__(self): 
        self.epoch = 0

    def on_epoch_end(self, model):
        self.epoch += 1
        print('\tIteration (Epoch)', str(self.epoch) )

# Train the word2vec model with CBOW
def train_word2vec(dataset_dir):

    lang = 'en'
    embed_file = os.path.join(dataset_dir, "embeddings_" + lang + ".vec")

    print('\n **** Acquiring the cleaned corpus for Sense Embeddings.. **** \n')
    # Get final pre-processed corpus
    TotDocs_expected = 20
    if len(os.listdir(corpus_file)) != TotDocs_expected:
        getData_Embed_Train(dataset_dir, lang)

    corpus = IterableCorpus(corpus_file)

    print('\n **** Training the data for Sense Embeddings.. **** \n')

    # Build the vocabulary from the corpus
    cores = multiprocessing.cpu_count() # Count the number of cores in a computer
    t = time.time()

    # Epoch logger
    iter_logger =IterLogger()

    # sg = 0 is for cbow (1 for skip-gram)
    w2v_model = Word2Vec(corpus,
                        min_count=3,
                        window=5,
                        size=400,
                        sample=6e-5,
                        alpha=0.03,
                        min_alpha=0.0007,
                        negative=30,
                        workers=cores-1,
                        cbow_mean=1,
                        iter=30,
                        callbacks=[iter_logger])


    print('\n **** Training the model for Sense Embeddings.. **** \n')

    print('Time spent training the model: {} mins'.format(round((time.time() - t) / 60, 2)))

    w2v_model.wv.save_word2vec_format(embed_file, binary=False)
    vocab = list((w2v_model.wv.vocab).keys())

    std, correlation = cosine_spearman_correlation(wordsim_file, w2v_model, vocab)

    return()

# Load the trained sense embeddings
def getTrained_SensEmbedings(embed_file):
    lemma_bn_embedding = []

    with open(embed_file) as file:
        embeddings = file.readlines()
        for info in embeddings:
            if "_bn" in info:
                lemma_bn_embedding.append(info)

    return(lemma_bn_embedding)

# Reload the trained word2vec model and vectors for vocabulary senses
def load_w2v_model(wordsim_file):
    from gensim.models import KeyedVectors
    vocabulary = []

    model = KeyedVectors.load_word2vec_format(embed_file, binary = False)
    vocab = list((model.vocab).keys())

    # vocabulary list of all word senses (excludes those that don't have any)
    for word in vocab:
        vocabulary.append(word)

    return(model, vocabulary)

# Calculate the accuracy of the model with word similarities
def cosine_spearman_correlation(wordsim_file, model, vocabulary):
    import scipy

    std = pd.read_csv(wordsim_file, delimiter = '\t')
    std['cosine'] = std.apply(lambda entry: word_similarity(entry['Word 1'],entry['Word 2'], model, vocabulary), axis=1)
    correlation, pvalue = scipy.stats.spearmanr(std['Human (mean)'], std['cosine'])
    std.to_csv (EvalOut_SensEmbed_file, index = None, header=True, sep='\t')
    print(correlation, pvalue)


    return(std, correlation)

# Calculates the similarity score between two words
def word_similarity(w1, w2, model, vocabulary):
    w1_senses = get_word_vector(w1,vocabulary)
    w2_senses = get_word_vector(w2,vocabulary)
    score = -1.0


    if len(w1_senses) > 0 and len(w2_senses) > 0:
        comb = itertools.product(w1_senses, w2_senses)
        for v1, v2 in comb:
            score = max(score, model.wv.similarity(v1, v2))

    return(score)

# Gets the embeddings of a word sense
def get_word_vector(word, vocabulary):
    wordSenses = []

    for w in vocabulary:
        Word = w.split("_")[0]
        if word.lower() == Word.lower():
            wordSenses.append(w)

    return(wordSenses)


train_word2vec(dataset_dir)
