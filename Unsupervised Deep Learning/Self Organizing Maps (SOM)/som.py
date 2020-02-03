# developing SOM

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import bone, pcolor, colorbar, plot, show

# Import the dataset
dataset = pd.read_csv(f'./Self_Organizing_Maps/Credit_Card_Applications.csv')
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
# print(mappings)
# getting customers according to the coordinates
frauds = np.concatenate((mappings[(8, 1)], mappings[(6, 8)]), axis=0)
# print(frauds)
frauds = sc.inverse_transform(frauds)
print(frauds)
