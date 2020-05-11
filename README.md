Abstract

A Named Entity Recognition (NER) task is
implemented to understand words as a loca-
tion (LOC), an organization (ORG), a person
(PER), or other (O). The neural network ar-
chitecture has a pre-trained Sense Embedding
layer, two Bi-LSTM layers with dropout, and
Adam optimizer. The model resulted in a F1
score of 0.806 for all classes.

Introduction
Language for an adult that understands the dialect
is simple and effortless. However in order for ma-
chines to understand this, Natural Language Pro-
cessing is necessary to be able to read, understand,
and derive meaning to language. One way to derive
meaning is by processing text using Named Entity
Recognition where the machine can recognize cer-
tain words as a location (LOC), an organization
(ORG), a person (PER), or other (O).
A NER neural network algorithm is used with a
Bidirectional Long short Term Memory (Bi-LSTM)
architecture and an embedding layer previously
trained using Sense Tags. The Bi-LSTM uses past
and future labels in a sequence to identify and clas-
sify named entities in text that can be later used for
tasks such as recommendation systems.

Diagram shows a 2D vector space of 100
words from the Sense embeddings. It was transformed
from a 400 to 2 dimension vector using only 2,000
words in the vocabulary to reduce computation com-
plexity. Words such as ’parliment’, ’country’, and ’eu’
are found in close proximity showing correctly their
similarity.
![](hw1/stud/images/embed_vector_1.png)

The table shows some of the main ex-
periments done for the model details such as adding
Sense embeddings, leaving capitalizations as is, tag-
ging words with their Senses given its most common
or WSD synset. Explanation of the reasoning behind
each chosen experiment is given. Those experiments
highlighted in green are the main model changes, and
the orange is the final chosen model.

![](hw1/stud/images/Table_model_exper.png)

The table shows the performances for the
chosen NER model. The final F1 score for the test
dataset is 0.806, which is close to the best value at 1
where both recall and precision are perfect. Recall is
the main issue with 0.788 meaning there are lower per-
centage of correct relevant results. The class that has
the worst F1 score is ’ORG’, as expected since this
class can contain words that usually on their own would
be labelled as any other class, making it the hardest to
learn (e.g. Expected: ’University of New York’ → all
’ORG’. Predicted : ’University’ → ’ORG’, ’of’ → ’O’,
’New York’ → both ’LOC’).

![](hw1/stud/images/Final_Performance_table_NER.png)
