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

vocab_size = 10000
embedding_dim = 128
max_length = 250
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
training_size = 16000 #5000=0.88,7500=0.92,8000=0.92
num_epochs = 10

df = pd.read_csv("korrigierte_trainingsdaten_ohne_germeval2018.csv",encoding="utf8",sep=";")
print(df.shape)

text = df["Tweets"].values.tolist()
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


labels = df['Hatespeech'].values.tolist()

text = newtext

tokenizer = Tokenizer(vocab_size)
tokenizer.fit_on_texts(text)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(text)

padded = pad_sequences(sequences, padding='post',maxlen=250)

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

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.005), metrics=['accuracy'])

history = model.fit(training_padded, training_labels, epochs=num_epochs,
                    validation_data=(testing_padded, testing_labels), verbose=2,batch_size=128)


df2 = pd.read_csv('germeval2018.training.txt', encoding="utf-8", sep='\t', names=('TEXT', 'TRASH1', 'TRASH2'))
labelsTRAINING = df2['TRASH1'].values.tolist()
df2.drop('TRASH1', axis=1, inplace=True)
df2.drop('TRASH2', axis=1, inplace=True)

trainingListe = df2["TEXT"].values.tolist()


sequences = tokenizer.texts_to_sequences(trainingListe)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
counterHATESPEECHinFILE = 0
counterforRESULTS = 0
for labels in labelsTRAINING:
    if labels != "OTHER":
        counterHATESPEECHinFILE = counterHATESPEECHinFILE + 1

p = model.predict(padded)
print(p)

for x in range(len(p)):
    print(str(p[x]) + "----" + str(labelsTRAINING[x]))
    if p[x] > 0.6 and labelsTRAINING[x] != 'OTHER':
        counterforRESULTS = counterforRESULTS + 1
    if p[x] <= 0.6 and labelsTRAINING[x] == 'OTHER':
        counterforRESULTS = counterforRESULTS + 1

print(len(labelsTRAINING))
print(counterforRESULTS)
print(counterforRESULTS / len(labelsTRAINING))
