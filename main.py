import csv

import pandas as pd
import tensorflow_hub as hub
import tensorflow_text
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras
from tempfile import TemporaryFile
import random
from tqdm import tqdm

use = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")

df3 = pd.read_csv("Testdaten.csv", encoding="utf8", sep=";")

# X_testdaten = []
# for r in tqdm(df3["c_text"].values.tolist()):
#     emb = use(r)
#     review_emb = tf.reshape(emb, [-1]).numpy()
#     X_testdaten.append(review_emb)
# X_testdaten = np.array(X_testdaten)
# np.save("X_testdaten.npy", X_testdaten)

RANDOM_SEED = 42

# df = pd.read_csv("Daten_ohne_germeval2018.csv", encoding="utf8", sep=";")
#
# type_one_hot = OneHotEncoder(sparse=False).fit_transform(
#     df.Hatespeech.to_numpy().reshape(-1, 1)
# )
#
# train_reviews, test_reviews, y_train, y_test = \
#     train_test_split(
#         df.Tweets,
#         type_one_hot,
#         test_size=.1,
#         random_state=RANDOM_SEED
#     )
# ---------Numpy Datei speichern
# X_train = []
# for r in tqdm(train_reviews):
#   emb = use(r)
#   review_emb = tf.reshape(emb, [-1]).numpy()
#   X_train.append(review_emb)
# X_train = np.array(X_train)
# np.save("X_train_Daten_ohne@_mit_stopwörter_50_50.npy",X_train)
# X_test = []
# for r in tqdm(test_reviews):
#   emb = use(r)
#   review_emb = tf.reshape(emb, [-1]).numpy()
#   X_test.append(review_emb)
# X_test = np.array(X_test)
# np.save("X_test_Daten_ohne@_mit_stopwörter_50_50.npy",X_test)
# np.save("Y_train_Daten_ohne@_mit_stopwörter_50_50.npy",y_train)
# np.save("Y_test_Daten_ohne@_mit_stopwörter_50_50.npy",y_test)

X_train = np.load("X_train_Daten_ohne@_mit_stopwörter_50_50.npy")
X_test = np.load("X_test_Daten_ohne@_mit_stopwörter_50_50.npy")
y_train = np.load("Y_train_Daten_ohne@_mit_stopwörter_50_50.npy")
y_test = np.load("Y_test_Daten_ohne@_mit_stopwörter_50_50.npy")

model = keras.Sequential()

model.add(
    keras.layers.Dense(
        units=256,
        input_shape=(X_train.shape[1],),
        activation='relu'
    )
)
model.add(
    keras.layers.Dropout(rate=0.5)
)

model.add(
    keras.layers.Dense(
        units=128,
        activation='relu'
    )
)
model.add(
    keras.layers.Dropout(rate=0.5)
)

model.add(keras.layers.Dense(2, activation='softmax'))
model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.Adam(0.001),
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    epochs=8,
    batch_size=32,
    validation_split=0.1,
    verbose=1,
    shuffle=True
)

X_testdaten = np.load("X_testdaten.npy")
X_wordlist = np.load("X_wordlist.npy")

counterHATESPEECHinModel = 0

p = model.predict(X_testdaten)

threshold = 0.45
predictions = []

for x in range(len(p)):
    if p[x][1] > threshold:
        counterHATESPEECHinModel = counterHATESPEECHinModel + 1
        predictions.append(1)
    if p[x][1] <= threshold:
        predictions.append(0)
    # if threshold > p[x][1] > threshold - 0.2:
    #     for word in X_wordlist:
    #         if word in X_testdaten[x]:
    #             counterHATESPEECHinModel = counterHATESPEECHinModel + 1
    #             predictions.append(1)
    #             break
    #     predictions.append(0)

threshold = threshold + 0.01

df3["hatespeech"] = predictions

df = pd.read_csv("Entwicklungsdatenfertig.csv", encoding="utf8", sep=";")

type_one_hot = OneHotEncoder(sparse=False).fit_transform(
    df.toxi.to_numpy().reshape(-1, 1)
)

train_reviews, test_reviews, y_train, y_test = \
    train_test_split(
        df.c_text,
        type_one_hot,
        test_size=.2,
        random_state=RANDOM_SEED
    )
# # ---------Numpy Datei speichern
# X_train = []
# for r in tqdm(train_reviews):
#   emb = use(r)
#   review_emb = tf.reshape(emb, [-1]).numpy()
#   X_train.append(review_emb)
# X_train = np.array(X_train)
# np.save("X_train_Entwicklungsdaten_ohne@.npy",X_train)
# X_test = []
# for r in tqdm(test_reviews):
#   emb = use(r)
#   review_emb = tf.reshape(emb, [-1]).numpy()
#   X_test.append(review_emb)
# X_test = np.array(X_test)
# np.save("X_test_Entwicklungsdaten_ohne@.npy",X_test)
# np.save("Y_train_Entwicklungsdaten_ohne@.npy",y_train)
# np.save("Y_test_Entwicklungsdaten_ohne@.npy",y_test)

X_train = np.load("X_train_Entwicklungsdaten_ohne@.npy")
X_test = np.load("X_test_Entwicklungsdaten_ohne@.npy")
y_train = np.load("Y_train_Entwicklungsdaten_ohne@.npy")
y_test = np.load("Y_test_Entwicklungsdaten_ohne@.npy")

print(X_train.shape, y_train.shape)

model = keras.Sequential()

model.add(
    keras.layers.Dense(
        units=256,
        input_shape=(X_train.shape[1],),
        activation='relu'
    )
)
model.add(
    keras.layers.Dropout(rate=0.5)
)

model.add(
    keras.layers.Dense(
        units=128,
        activation='relu'
    )
)
model.add(
    keras.layers.Dropout(rate=0.5)
)

model.add(keras.layers.Dense(5, activation='softmax'))
model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.Adam(0.001),
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    epochs=8,
    batch_size=32,
    validation_split=0.2,
    verbose=1,
    shuffle=True
)

p = model.predict(X_testdaten)
toxi = []

for x in range(len(p)):
    toxi.append(np.argmax(p[x])+1)

df3["toxi"] = toxi
df3.to_csv("Testdaten_mit_Predictions.csv", encoding="utf8", sep=";")

counter1 = 0
counter2 = 0
counter3 = 0
counter4 = 0
counter5 = 0

for x in range(len(p)):
    max = np.argmax(p[x])

    if max == 0:
        counter1 = counter1 + 1
    if max == 1:
        counter2 = counter2 + 1
    if max == 2:
        counter3 = counter3 + 1
        print(p[x])
    if max == 3:
        counter4 = counter4 + 1
    if max == 4:
        counter5 = counter5 + 1

print(str(counter1))
print(str(counter2))
print(str(counter3))
print(str(counter4))
print(str(counter5))
