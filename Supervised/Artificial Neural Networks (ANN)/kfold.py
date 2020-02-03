# importing the required libraries
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from tensorflow import keras
import tensorflow as tf
import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Flatten, Dropout
# from tensorflow.keras import datasets, layers, models
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# Importing dataset
dataset = pd.read_csv(
    r"C:\Users\ChampWk38\Desktop\Deep_Learning_A_Z\Volume 1 - Supervised Deep Learning\Part 1 - Artificial Neural Networks (ANN)\Section 4 - Building an ANN\Artificial_Neural_Networks\Churn_Modelling.csv")
X = dataset.iloc[:, 3: 13].values
y = dataset.iloc[:, 13].values

# Encoding the categorical data
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = ColumnTransformer(transformers=[('Test', OneHotEncoder(), [1])], remainder='passthrough')
# onehotencoder = OneHotEncoder(categories="auto")
X = onehotencoder.fit_transform(X)
# print(X[0])
X = X[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


#
# # fitting the ANN to the training
# classifier.fit(X_train, y_train, batch_size=10, epochs=100, verbose=2)
#
# # making the predictions and evaluation
# y_pred = classifier.predict(X_test)
#
# y_pred = (y_pred > 0.5)
#
# cm = confusion_matrix(y_test, y_pred)
#
# print(cm)
# acc = (cm[0][0] + cm[1][1]) / 2000
# print(acc)

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(output_dim=6, init="uniform", activation="relu", input_dim=11))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(output_dim=6, init="uniform", activation="relu"))
    classifier.add(Dropout(rate=0.1))
    classifier.add(Dense(output_dim=1, init="uniform", activation="sigmoid"))
    classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return classifier


classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=100)
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=-1)
mean = accuracies.mean()
variance = accuracies.std()

print(mean)
print(variance)
