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

RANDOM_SEED = 42

use = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")

df = pd.read_csv("Daten_ohne_germeval2018.csv", encoding="utf8", sep=";")

type_one_hot = OneHotEncoder(sparse=False).fit_transform(
    df.Hatespeech.to_numpy().reshape(-1, 1)
)

train_reviews, test_reviews, y_train, y_test = \
    train_test_split(
        df.Tweets,
        type_one_hot,
        test_size=.1,
        random_state=RANDOM_SEED
    )
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

df2 = pd.read_csv('germeval2018.training.txt', encoding="utf-8", sep='\t', names=('TEXT', 'TRASH1', 'TRASH2'))
labelsTRAINING = df2['TRASH1'].values.tolist()
labelsTRAINING2 = df2['TRASH2'].values.tolist()
df2.drop('TRASH1', axis=1, inplace=True)
df2.drop('TRASH2', axis=1, inplace=True)

# X_germ = []
# for r in tqdm(df2["TEXT"].values.tolist()):
#   emb = use(r)
#   review_emb = tf.reshape(emb, [-1]).numpy()
#   X_germ.append(review_emb)
# np.save("X_germ_2018.npy",X_germ)

X_germ = np.load("X_germ_2018.npy")

counterHATESPEECHinFILE = 0
counterHATESPEECHinModel = 0
counterHATESPEECHinSum = 0
counterforRESULTS = 0
for i in range(len(labelsTRAINING)):
    if labelsTRAINING[i] != "OTHER" and labelsTRAINING2[i] != "PROFANITY":
        counterHATESPEECHinFILE = counterHATESPEECHinFILE + 1

p = model.predict(X_germ)

threshold = 0.1

for i in range(90):
    print("-----------" + str(threshold) + "-------------")
    counterHATESPEECHinModel = 0
    counterforRESULTS = 0
    counterHATESPEECHinSum = 0
    for x in range(len(p)):
        if p[x][1] > threshold and labelsTRAINING[x] != 'OTHER' and labelsTRAINING2[i] != "PROFANITY":
            counterHATESPEECHinModel = counterHATESPEECHinModel + 1
            counterforRESULTS = counterforRESULTS + 1
        if p[x][1] <= threshold and labelsTRAINING[x] == 'OTHER' or labelsTRAINING2[i] == "PROFANITY":
            counterforRESULTS = counterforRESULTS + 1
        if p[x][1] > threshold:
            counterHATESPEECHinSum = counterHATESPEECHinSum + 1

    precision = counterHATESPEECHinModel / counterHATESPEECHinSum
    recall = counterHATESPEECHinModel / counterHATESPEECHinFILE

    print("Hatespeech in File: " + str(counterHATESPEECHinFILE))
    print("Richtige Hatespeech in Model: " + str(counterHATESPEECHinModel))
    print("Alle Hatespeech in Model: " + str(counterHATESPEECHinSum))
    print("Hatespeechhit / Recall :" + str(recall))
    print("Insgesamte Labels: " + str(len(labelsTRAINING)))
    print("Richtige Labels in Model: " + str(counterforRESULTS))
    print("Insgesmte Hitrate / Accuracy :" + str(counterforRESULTS / len(labelsTRAINING)))
    print("Precision: " + str(precision))
    if precision + recall > 0:
        print(("F: " + str(2 * ((precision * recall) / (precision + recall)))))
    threshold = threshold + 0.01
