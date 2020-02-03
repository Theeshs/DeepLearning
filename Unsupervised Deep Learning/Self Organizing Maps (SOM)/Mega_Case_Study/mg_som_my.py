# Maga Case Study - Build HyBrid Fraud Detection

# Part 1 - developing SOM

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from minisom import MiniSom
from pylab import bone, pcolor, colorbar, plot, show
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense

# Import the dataset
dataset = pd.read_csv(f'Credit_Card_Applications.csv')
X = dataset.iloc[:, :-1].values
# print(X)
y = dataset.iloc[:, -1].values
# print(y)

# Feature Scaling
sc = MinMaxScaler(feature_range=(0, 1))
X = sc.fit_transform(X)

# Training SOM
som = MiniSom(x=10, y=10, input_len=15, sigma=1.0, learning_rate=0.5)
som.random_weights_init(X)
som.train_random(data=X, num_iteration=100)
# som.save()
# Visualizing the results
bone()
pcolor(som.distance_map().T)
colorbar()
markers = ['o', 's']
colors = ['r', 'g']

for i, x in enumerate(X):
    w = som.winner(x)
    plot(w[0] + 0.5,
         w[1] + 0.5,
         markers[y[i]],
         markeredgecolor=colors[y[i]],
         markerfacecolor='None',
         markersize=10,
         markeredgewidth=2)
show()

# Finding the frauds
mappings = som.win_map(X)
print(mappings)
# getting customers according to the coordinates
# exit()
frauds = np.concatenate((mappings[(8, 1)], mappings[(6, 8)]), axis=0)
frauds = sc.inverse_transform(frauds)

# Part 2 - Supervising deep learning model

# create the matrix feature
customers = dataset.iloc[:, 1:].values

# Creating the dependent variable
is_fraud = np.zeros(len(dataset))
for i in range(len(dataset)):
    if dataset.iloc[i, 0] in frauds:
        is_fraud[i] = 1

# Feature Scaling
sc = StandardScaler()
customers = sc.fit_transform(customers)

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units=2, kernel_initializer='uniform', activation='relu', input_dim=15))

# Adding the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(customers, is_fraud, batch_size=1, epochs=2)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(customers)
y_pred = np.concatenate((dataset.iloc[:, 0:1].values, y_pred), axis=1)
print(y_pred)
y_pred = y_pred[y_pred[:, 1].argsort()]
