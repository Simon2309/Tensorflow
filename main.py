import pandas as pd
import tensorflow_hub as hub
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras
from tqdm import tqdm

# ----------------------------------------------------------------------------------------------------------------------------------
# Der Codeabschnitt bis zu nächsten Trennlinie ist nur zum laden und vorbereiten der Daten.
# Dieser Schritt muss nur einmal ausgeführt werden da dann die Daten in den Numpy Datein gespeichert sind

# Der Universal Sentence Encoder codiert Sätze in einen embedding vektor mit 512 Elementen unabhängig von der länge des Satzes
# Zudem wird eine Vorverabreitung der Daten von dem Model durchgeführt. Die Vorverarbeitung umfasst z.B. das entfernen von Stopwörter und Tokenisierung
use = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")

# Einlesen der Testdaten in ein Panda Datenframe mit dem Encoding utf-8 und den Trennzeichen ";"
df3 = pd.read_csv("Testdaten.csv", encoding="utf8", sep=";")

# Testdaten werden mithilfe von dem geladenen Model in ein Vektor umgewandelt. Das Numpyarray wird abgespeichert.
X_testdaten = []
for r in tqdm(df3["c_text"].values.tolist()):
    emb = use(r)
    review_emb = tf.reshape(emb, [-1]).numpy()
    X_testdaten.append(review_emb)
X_testdaten = np.array(X_testdaten)
np.save("X_testdaten.npy", X_testdaten)

# Integer um beim Splitten der Daten durchzumischen
RANDOM_SEED = 42

# Einlesen der Daten die zum trainieren des Models genutzt werden.Encoding utf-8 und den Trennzeichen ";"
df = pd.read_csv("Daten_ohne_germeval2018.csv", encoding="utf8", sep=";")

# Vektor der Hatespeechlabels wird umgewandelt um es mit den Tweetvektor kompatibel zu machen
type_one_hot = OneHotEncoder(sparse=False).fit_transform(
    df.Hatespeech.to_numpy().reshape(-1, 1)
)

# Aufteilung der eingelesen Daten in Training- und Testdaten mit dem RANOM_SEED zum Vermischen und der test_size=0.1.
# Das heißt 10% der Daten werden zum Testen des Models verwendet
train_reviews, test_reviews, y_train, y_test = \
    train_test_split(
        df.Tweets,
        type_one_hot,
        test_size=.1,
        random_state=RANDOM_SEED
    )
# Trainingsdaten vom Model werden mithilfe von dem geladenen Model in ein Vektor umgewandelt. Das Numpyarray wird abgespeichert.
#In den X Arrays stehen die Tweets und in dem Y Arrays stehen die Hatespeechlabels
X_train = []
for r in tqdm(train_reviews):
    emb = use(r)
    review_emb = tf.reshape(emb, [-1]).numpy()
    X_train.append(review_emb)
X_train = np.array(X_train)
# Testdaten vom Model werden mithilfe von dem geladenen Model in ein Vektor umgewandelt. Das Numpyarray wird abgespeichert.
X_test = []
for r in tqdm(test_reviews):
    emb = use(r)
    review_emb = tf.reshape(emb, [-1]).numpy()
    X_test.append(review_emb)
X_test = np.array(X_test)
# Abspeichern der Numpy Arrays damit dei Daten schnell neu geladen werden können.
np.save("X_train_Daten.npy", X_train)
np.save("X_test_Daten.npy", X_test)
np.save("Y_train_Daten.npy", y_train)
np.save("Y_test_Daten.npy", y_test)

# ----------------------------------------------------------------------------------------------------------------------------------------------
#Der nächste Codeabschnitt ist das Model für die Binärentscheidung ob es Hatespeech ist oder nicht

# Laden der Arrays aus den Datein
X_train = np.load("X_train_Daten.npy")
X_test = np.load("X_test_Daten.npy")
y_train = np.load("Y_train_Daten.npy")
y_test = np.load("Y_test_Daten.npy")

# Erstellung eines Sequential Model mit einem linearen Stack von Layern
model = keras.Sequential()

# 1. Layer:  Dense Layer mit 256 Units. In einem Dense Layer sind alle Neuronen untereinander verbunden
model.add(
    keras.layers.Dense(
        units=256,
        input_shape=(X_train.shape[1],),
        activation='relu'
    )
)

# 2. Dropoutlayer der Overfitting des Models verhindert
model.add(
    keras.layers.Dropout(rate=0.5)
)

# 3. Neuer Dense Layer mit der Hälfte an Units
model.add(
    keras.layers.Dense(
        units=128,
        activation='relu'
    )
)
# 4. Dropoutlayer der Overfitting des Models verhindert
model.add(
    keras.layers.Dropout(rate=0.5)
)

# 5. OutputLayer mit nur 2 Units für Binärentscheidung ob es Hatespeech ist oder nicht
model.add(keras.layers.Dense(2, activation='softmax'))

# Übersetzen des Models
model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.Adam(0.001),
    metrics=['accuracy']
)

#Das Model wird mit den Trainings- und Testdaten trainiert. Es werden 8 Durchläufe gemacht.Aufteilung der Testdaten sind 10%.
# Und es wird nach jedem Durchlauf neu gemischt
model.fit(
    X_train, y_train,
    epochs=8,
    batch_size=32,
    validation_split=0.1,
    verbose=1,
    shuffle=True
)

#Laden des Testdatenarrays
X_testdaten = np.load("X_testdaten.npy")


#Model predicted die Daten
p = model.predict(X_testdaten)

#Grenzwert bei dem das Model entscheidet wann es Hatespeech erkennt.
# Wenn die Wahrscheinlichkeit 45% Hatespeech beträgt wird der Tweet als Hatespeech erkannt
threshold = 0.45

#Schleife über das Ergebnisarray von dem Model. An der Stelle 1 steht die Wahrscheinlichkeit für Hatespeech
predictions = []
for x in range(len(p)):
    if p[x][1] > threshold:
        predictions.append(1)
    if p[x][1] <= threshold:
        predictions.append(0)

#Speichern der Predictions in das Panda Dataframe
df3["hatespeech"] = predictions

#-----------------------------------------------------------------------------------------------------------------------
#Bei dem Model zur Erkennung des Toxizität Levels wird das gleiche Model benutzt.

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
X_train = []
for r in tqdm(train_reviews):
  emb = use(r)
  review_emb = tf.reshape(emb, [-1]).numpy()
  X_train.append(review_emb)
X_train = np.array(X_train)
np.save("X_train_Entwicklungsdaten.npy",X_train)
X_test = []
for r in tqdm(test_reviews):
  emb = use(r)
  review_emb = tf.reshape(emb, [-1]).numpy()
  X_test.append(review_emb)
X_test = np.array(X_test)
np.save("X_test_Entwicklungsdaten.npy",X_test)
np.save("Y_train_Entwicklungsdaten.npy",y_train)
np.save("Y_test_Entwicklungsdaten.npy",y_test)

X_train = np.load("X_train_Entwicklungsdaten.npy")
X_test = np.load("X_test_Entwicklungsdaten.npy")
y_train = np.load("Y_train_Entwicklungsdaten.npy")
y_test = np.load("Y_test_Entwicklungsdaten.npy")

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

#5 Outputlayer. Ein Layer für jedes Level
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
#Pro Tweet stehen jetzt 5 Wahrscheinlichkeiten in dem Ergebnisarray. Hier wird geschaut welches Level am wahrscheinlichsten ist.
# Die Stelle vom größten Wert in dem Array wird genommen
for x in range(len(p)):
    toxi.append(np.argmax(p[x]) + 1)

#Abspeichern in das Panda Dataframe
df3["toxi"] = toxi

#Panda Dataframe als CSV abspeichern
df3.to_csv("Testdaten_mit_Predictions.csv", encoding="utf8", sep=";")
