# import csv
import csv
import re
import string

import pandas as pd
import tensorflow as tf
import tokenizer as tokenizer

from tensorflow import keras
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

import numpy as np
import random
import spacy

vocab_size = 1000
embedding_dim = 16
max_length = 50
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
training_size = 6000
num_epochs = 10
classes = ["No Hatespeech", "Hatespeech"]
class_to_index = dict((c, i) for i, c in enumerate(classes))
index_to_Class = dict((v, k) for k, v in class_to_index.items())


df = pd.read_csv('trainingsdaten.csv', encoding="utf-8", sep=';')

df2 = pd.read_csv("Daten.csv", encoding="utf-8", sep=";")

df.drop('c_id', axis=1, inplace=True)
df.drop('date', axis=1, inplace=True)
df.drop('author_id', axis=1, inplace=True)
df.drop('like_count', axis=1, inplace=True)
df.drop('quote_count', axis=1, inplace=True)
df.drop('retweet_count', axis=1, inplace=True)
df.drop('reply_count', axis=1, inplace=True)
df.drop('toxi', axis=1, inplace=True)
df.drop('target', axis=1, inplace=True)

text = df['c_text'].values.tolist()
newtext = []

for row in text:
    list = str(row).split(" ")
    for word in list:
        if word.startswith("@"):
            list.remove(word)
    for word in list:
        if word.startswith("@"):
            list.remove(word)
    for word in list:
        if word.startswith("@"):
            list.remove(word)
    newtext.append(" ".join(list))

# for row in df2["text"].values:
#     newtext.append(row)

labels = df['hatespeech'].values.tolist()

# for row in df2["label"]:
#     labels.append(int(row))

nlp = spacy.load("de_core_news_lg")

lemmalist = []

# ---------------------Lemma Liste Abspeichern
# for row in newtext:
#     doc = nlp(row)
#     list = []
#     for token in doc:
#         list.append(token.lemma_.lower())
#     lemmalist.append(" ".join(list))
# lemmalistclean = []
#
# for row in lemmalist:
#     text = re.sub('https?://\S+|www.\S+', '', row)
#     text = re.sub('\n', '', text)
#     text = re.sub('\r', '', text)
#     lemmalistclean.append(text)
# print(lemmalistclean)
# dict = {"Text": lemmalistclean}
# df = pd.DataFrame(dict)
# df.to_csv("Lemmalist3.csv")
# -----------------------------------


# df = pd.read_csv('Lemmalist3.csv', encoding="utf-8", sep=',')
# df.drop('Unnamed: 0', axis=1, inplace=True)
#
# text = []
#
# for row in df['Text'].values.tolist():
#     text.append(str(row))

text = newtext

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(text)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(text)

padded = pad_sequences(sequences, padding='post')
# print(padded[0])
# print(padded.shape)

training_sentences = text[0:training_size]
testing_sentences = text[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

# model = tf.keras.Sequential([
#     tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
#     tf.keras.layers.GlobalAveragePooling1D(),
#     tf.keras.layers.Dense(4, activation='relu'),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(10000, 16, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20, return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(20)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

history = model.fit(training_padded, training_labels, epochs=num_epochs,
                    validation_data=(testing_padded, testing_labels), verbose=2)

# trainingsDaten =pd.read_csv('germeval2018.training.txt', encoding='utf8', sep='\t')

df2 = pd.read_csv('germeval2018.training.txt', encoding="utf-8", sep='\t', names=('TEXT', 'TRASH1', 'TRASH2'))
labelsTRAINING = df2['TRASH1'].values.tolist()
df2.drop('TRASH1', axis=1, inplace=True)
df2.drop('TRASH2', axis=1, inplace=True)

# df2 = pd.read_csv("GermevalLemma.csv", encoding="utf-8", sep=",")
# df2.drop('Unnamed: 0', axis=1, inplace=True)

# trainingListe = df2["Text"].values.tolist()
trainingListe = df2["TEXT"].values.tolist()


sequences = tokenizer.texts_to_sequences(trainingListe)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
counterHATESPEECHinFILE = 0
counterforRESULTS = 0
for labels in labelsTRAINING:
    if labels != "OTHER":
        counterHATESPEECHinFILE = counterHATESPEECHinFILE + 1

# results = model.predict(padded)
# p = model.predict(np.expand_dims(padded, axis=0))
p = model.predict(padded)
print(p)



for x in range(len(p)):
    print(str(p[x]) + "----" + str(labelsTRAINING[x]))

for x in range(len(p)):
    if p[x] > 0.5:
        print('1' + '  <------>  ' + labelsTRAINING[x])

    else:
        print('0' + '  <------>  ' + labelsTRAINING[x])
    if p[x] > 0.5 and labelsTRAINING[x] != 'OTHER':
        counterforRESULTS = counterforRESULTS + 1
    if p[x] <= 0.5 and labelsTRAINING[x] == 'OTHER':
        counterforRESULTS = counterforRESULTS + 1

print(len(labelsTRAINING))
print(counterforRESULTS)
print(counterforRESULTS / len(labelsTRAINING))
