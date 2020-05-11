"""
Author: Cecilia Aponte
Natural Language Processing
Word POS Embeddings

Pre-Processing SenseEmbed
"""

# Import librarie=s
import os
import re, string 
from nltk.corpus import stopwords
from lxml import etree
from keras.preprocessing import text
import tarfile, time, pickle


punctuationALL = string.punctuation
punctuation = string.punctuation
punctuation = punctuation.replace("-", "") # don't remove hyphens
punctuation = punctuation.replace("'", "") # don't remove aposterphe
stopwords = []


# Get all the EuroSense data to preprocess for Sense Embeddings
def getData_Embed_Train(dataset_dir, lang):

    TargetLemmasVocab = getVocab_Embed_Train(dataset_dir, lang)
    bn2wn = check_bn2wn(dataset_dir)

    train_file = tarfile.open(os.path.join(dataset_dir,"SenseEmbed_Training_Corpora/eurosense.v1.0.high-precision.tar.gz"), encoding='utf-8')

    for file in train_file.getmembers():
        train_data = (train_file.extractfile(file))

    TargetSentences = ''
    SentTot_expected = 1903116
    QtySentences = 0
    TotDocs_expected = 2
    TotDocs_curr = 0
    QtySentperDoc = 100000
    TotDocs_expected = (SentTot_expected // QtySentperDoc) + 1

    print('\n **** Acquiring data to train SenseEmbed model.. **** \n')

    # Get an iterable
    corpora = etree.iterparse(train_data, events=('end',), tag='sentence')

    time_total = time.time()

    for _, item in corpora:
       Anchor_Lemma_bnID = []

       # Get the sentences of the corpora
       for sentence in item.iter():

           if QtySentences > -1: # run entire dataset

               for info in sentence.iter():
                   if info.tag == 'annotation' and info.attrib.get('lang') == lang and info.attrib.get('lemma') not in punctuationALL and info.attrib.get('lemma') in TargetLemmasVocab and info.text in bn2wn:
                       lemma = info.attrib.get('lemma').lower()
                       anchor = info.attrib.get('anchor').lower()
                       bn = info.text
                       Anchor_Lemma_bnID.append(anchor)
                       Anchor_Lemma_bnID.append(lemma + '_' + bn)

               if sentence.attrib.get('lang') == lang and sentence.text != None and sentence.tag == 'text':

                   QtySentences += 1
                   # Makes all words lowercase and strips of characters
                   sentenceStr = sentence.text.lower()
                   # Remove aposterphe naming
                   sentenceStr = re.sub(' &apos;s', '', sentenceStr)
                   sentenceStr = re.sub(' &apos;', '', sentenceStr)
                   # Removes all punctuation
                   sentenceStr = re.sub('['+punctuation+']', '', sentenceStr) #.split()
                   # Removes articles that don't add value to sense
                   sentenceStr = ' '.join([i for i in sentenceStr.split() if i not in stopwords])
                   # Remove extra blank spaces
                   sentenceStr = re.sub(' +', ' ', sentenceStr)
                   # Removes any set of digits to a single '#'
                   sentenceStr = re.sub('[-+]?[0-9]*\-?[0-9]+', '#', sentenceStr)
                   # Removes letters joined to words into single '#'
                   sentenceStr = re.sub(r'([a-z])+(?=[#]+)', '#', sentenceStr)
                   # Remove duplicate # to single one
                   sentenceStr = re.sub('# #', '#', sentenceStr)
                   sentenceStr = re.sub('# and #', '#', sentenceStr)
                   sentenceStr = re.sub('[#]+', '#', sentenceStr)
                   # Remove hypens that are not connecting to words
                   sentenceStr = re.sub(' - ', ' ', sentenceStr)


                   # Replace each anchor (actual word) with the lemma_bnID
                   if len(Anchor_Lemma_bnID) > 0:
                       sentSplit = sentenceStr.split()
                       for j, word in enumerate(sentSplit):
                           for i, anchor in enumerate(Anchor_Lemma_bnID[::2]):
                               if word == anchor:
                                   lemma_bnID = Anchor_Lemma_bnID[2*i+1]
                                   sentSplit[j] = lemma_bnID

                       sentenceStr =  ' '.join(sentSplit)
                   TargetSentences += sentenceStr + '\n'

                   # Save the target sentences in separate files to reduce memory issues
                   if QtySentences % QtySentperDoc == 0 or ((TotDocs_curr * QtySentperDoc) + len(TargetSentences.strip().split('\n')) == SentTot_expected):
                       TotDocs_curr += 1
                       TargetSentences = TargetSentences.strip()
                       with open(dataset_dir + "/SenseEmbed_Output_Processed/w2vCorpus/corpus_clean_" + str(TotDocs_curr) + '.pkl', 'wb+') as file:
                           pickle.dump(TargetSentences, file)
                       TargetSentences = ''
                       print(TotDocs_curr, 'of', TotDocs_expected, 'total documents saved with cleaned data from corpus.\n')

           else: break

        # Eliminate now-empty references from the root node to clear memory
       item.clear()
       while item.getprevious() is not None:
            del item.getparent()[0]


       # Check that the model is still running by printing a check
       if QtySentences % 3000 == 0:
           percent_done = str(format(QtySentences/SentTot_expected*100, '.2f')) + '%'
           print('Running pre-process.', percent_done, 'of total sentences processed and saved.')
           print('Total time elapsed pre-processing dataset: {} hrs'.format(round((time.time() - time_total) / 3600, 2)), '\n')

    print('All data pre-processed and saved.')

    print('Qty of sentences of initial corpus: ', QtySentences)
    print('Qty of sentences of mid corpus: ', len(TargetSentences))

    del corpora
    del TargetSentences


    return()

# Save or load the parsed corpus
def save_load_parsed(corpus, save, dataset_dir):
    import pickle
    saveParsed_file = os.path.join(dataset_dir,"SenseEmbed_Output_Processed/parsed_corpus")

    if save == True:
        outfile = open(saveParsed_file,'wb+')
        pickle.dump(corpus, outfile)
        outfile.close()
        parsed = 'Done saving parsed file'

    else:
        infile = open(saveParsed_file,'r')
        parsed = pickle.load(infile)
        infile.close()

    return(parsed)


# Get all Vocabulary annotations for Sense Embeddings
def getVocab_Embed_Train(dataset_dir, lang):

    print('\n **** Acquiring annotation vocabulary to train Sense Embeddings model.. **** \n')

    # If file already exists, load the data for TargetLemmasVoc
    try:
        with open(os.path.join(dataset_dir, 'SenseEmbed_Output_Processed/TargetLemmasVocabTxt.txt'), 'r') as output:
            TargetLemmasVocab = [line.strip().replace("'", '') for line in output]

        print(' -Target word vocabulary of size ', len(TargetLemmasVocab), ' created with annotations that appear at least 5 times in the dataset.')


    # Else, create the list from scratch
    except IOError:
        from collections import Counter
        TargetLemmasVocab = []

        train_file = tarfile.open(os.path.join(dataset_dir,"SenseEmbed_Training_Corpora/eurosense.v1.0.high-precision.tar.gz"), encoding='utf-8')

        for file in train_file.getmembers():
            train_data = (train_file.extractfile(file))

        # Get an iterable
        corpora = etree.iterparse(train_data, events=('end',), tag='sentence')

        # Get a list of all annotation words and check that they occur at least
        # 5 times so that it has enough examples to train
        for _, item in corpora:
            # Get the sentences of the corpora
            for sentence in item.iter():
                for info in sentence.iter():
                    if info.tag == 'annotation' and info.attrib.get('lang') == lang and info.attrib.get('lemma') not in punctuationALL and '@card@' not in info.attrib.get('lemma'):
                        lemma = info.attrib.get('lemma')
                        TargetLemmasVocab.append(lemma)

            item.clear()
            while item.getprevious() is not None:
                 del item.getparent()[0]
        print(' -Amount of total annotations in corpous is ', len(TargetLemmasVocab))
        TargetLemmasVocab = Counter(TargetLemmasVocab)
        # Remove annotations that occur less than 5 times
        TargetLemmasVocab = list(set([el for el in TargetLemmasVocab.elements() if TargetLemmasVocab[el] >= 5]))
        print(' -Target word vocabulary of size ', len(TargetLemmasVocab), ' created with annotations that appear at least 5 times in the dataset.')

        with open(os.path.join(dataset_dir, 'SenseEmbed_Output_Processed/TargetLemmasVocabTxt.txt'), 'w+') as output:
            for item in TargetLemmasVocab:
                output.write("%s\n" % item)

        del corpora

    return(TargetLemmasVocab)


# Check that an annotation with BN ID is in WordNet also (mapping of bn2wn)
def check_bn2wn(dataset_dir):
    import csv
    bn2wn = []

    with open(os.path.join(dataset_dir, 'babelnet2wordnet.tsv'), 'r')  as fileBn2Wn:
        rd = csv.reader(fileBn2Wn, delimiter="\t", quotechar='"')
        for row in rd:
            bn2wn.append(row[0])

    return(bn2wn)

# Get the synset in WordNet
def get_synWN():
    from nltk.corpus import wordnet as wn
    bn2wn = check_bn2wn(dataset_dir) # get the mappings

    for synID in bn2wn[1::2]:
        syn = wn.synset_from_pos_and_offset(synID[-1],int(synID[:-1]))

    return()

# Tokenized corpus after pre-processing where TargetSentences is a list of all
# sentences as strings and will be used directly into word2vec model
def tokenize_corpus(TargetSentences):

    final_corpus = []

    for sentence in TargetSentences:
        token_sent = text.text_to_word_sequence(sentence, filters='!"$%&()*+,./;<=>?@[\\]^`{|}~\t\n',split=' ')
        final_corpus.append(token_sent)

    return(final_corpus)


# Get all the semeval data for all languages
def getData_SenseEmb(dataset_dir):
    TargetWords = []
    print('\n **** Acquiring data to create sense embeddings.. **** \n')
    for filename in os.listdir(dataset_dir):
        if filename.endswith(".xml"):
            keyFile = ".".join(filename.split(".", 2)[:2]) + '.gold.key.txt'
            keyItems = []
            with open(os.path.join(dataset_dir, keyFile), "r") as file:
                for line in file.readlines():
                    line = line.split()[0]
                    keyItems.append(line)
            with open(os.path.join(dataset_dir, filename), "r") as file:
                data =  file.read()
                tree = ET.fromstring(data)
                print('Data from: \t', tree.attrib)
                instances = 0
                Corpus = []
                countTarget = []
                for paragraph in tree:
                    Paragraphs = []
                    for sentence in paragraph.iter('sentence'):
                        Sentences = []
                        instances += len(sentence.findall('instance'))
                        for word_item in sentence:
                            # remove any punctuations except -_~ and accented words or foreign letters
                            lemma = re.sub("[^[A-Z]\S+\s*- ]+",'', word_item.attrib['lemma'])
                            # removes words that are one or two letters of length
                            lemma = re.sub(r'¿', "", lemma)
                            pos = word_item.attrib['pos']

                            # when the word has a sense id
                            # if the first item is the id (meaning and instance) and in the Key file to converte to BabelNet ID
                            if list(word_item.attrib)[0] == 'id' and word_item.attrib['id'] in keyItems:
                                id = word_item.attrib['id']
                                word = [lemma, pos, id]
                                source = tree.attrib['source'].split('.')[0]
                                wordLang = [lemma, pos, id, source, tree.attrib['lang']]
                                TargetWords.append(wordLang)
                                countTarget.append(id)
                            else:
                                word = [lemma, pos]
                            # remove punctuation
                            if len(word[0]) > 1 or word[0] == 'y' or word[0] == 'e' or word[0] == 'è': # remove punctuation
                                Sentences.append(word) # add word to sentence
                        Paragraphs.append(Sentences) # add sentence to paragraph
                    Corpus.append(Paragraphs)
                print('Amount of target words for translation:\t', len(set(countTarget)))

    return(tree, TargetWords)


# For TESTING only
# bn2wn = check_bn2wn()
#tree, TargetWords = getData_SenseEmb(dataset_dir)
# TargetLemmas, TargetLemmasVocab = getData_Embed_Train(dataset_dir)
# final_corpus = getData_Embed_Train(dataset_dir)
# TargetLemmasVocab = getVocab_Embed_
