# Recurrent Neural Network

# Part 1 - Data prepossessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

# Import training set
dataset_train = pd.read_csv(
    r'C:\Users\ChampWk38\Desktop\Deep_Learning_A_Z\Volume 1 - Supervised Deep Learning\Part 3 - Recurrent Neural Networks (RNN)\Section 12 - Building a RNN\Recurrent_Neural_Networks\Google_Stock_Price_Train.csv')

training_set = dataset_train.iloc[:, 1:2].values

sc = MinMaxScaler(feature_range=(0, 1), copy=True)

training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timestamps and 1 output
X_train = []
y_train = []

for i in range(60, 1258):
    X_train.append(training_set_scaled[i - 60: i, 0])
    y_train.append(training_set_scaled[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
# print(X_train)

# Part 2 - Building the layers
# Initializing the RNN
def build_classifier(optimizer):
    regressor = Sequential()

    # Adding LSTM and Dropout Layers
    regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    regressor.add(Dropout(rate=0.2))

    # Adding second LSTM and Dropout
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(rate=0.2))

    # Adding third LSTM and Dropout
    regressor.add(LSTM(units=50, return_sequences=True))
    regressor.add(Dropout(rate=0.2))

    # Adding fourth LSTM and Dropout
    regressor.add(LSTM(units=50))
    regressor.add(Dropout(rate=0.2))

    # Adding output layers
    regressor.add(Dense(units=1))

    # Compiling the RNN
    regressor.compile(optimizer=optimizer, loss='mean_squared_error')
    return regressor

regressor = KerasClassifier(build_fn=build_classifier)

# Grid search implementation
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator=regressor,
                           param_grid=parameters,
                           scoring="accuracy",
                           cv=5,
                           n_jobs=1)

# Fitting the RNN to the training data
grid_search.fit(
    X_train,
    y_train
)

best_parameters = grid_search.best_estimator_
best_accuracy = grid_search.best_score_

print(best_parameters)
print(best_accuracy)

# Part 3 - Making the predictions and visualizing the results

# Getting the real stock prices of 2017
# dataset_test = pd.read_csv(
#     r'C:\Users\ChampWk38\Desktop\Deep_Learning_A_Z\Volume 1 - Supervised Deep Learning\Part 3 - Recurrent Neural Networks (RNN)\Section 12 - Building a RNN\Recurrent_Neural_Networks\Google_Stock_Price_Test.csv'
# )
# real_stock_price = dataset_test.iloc[:, 1:2].values
#
# # Getting the predicted stock prices of 2017
# dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
# inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
# inputs = inputs.reshape(-1, 1)
# inputs = sc.transform(inputs)
#
# X_test = []
#
# for i in range(60, 80):
#     X_test.append(inputs[i - 60: i, 0])
#
# X_test = np.array(X_test)
#
# X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
#
# predicted_stock_price = regressor.predict(X_test)
# predicted_stock_price = sc.inverse_transform(predicted_stock_price)
#
# # Visualizing the final results
# plt.plot(real_stock_price, color='red', label='Real Google Price')
# plt.plot(predicted_stock_price, color='blue', label='Predicted Google Price')
# plt.title('Google Stock Price Prediction')
# plt.xlabel('Time')
# plt.ylabel('Google Stock Price')
# plt.legend()
# plt.show()
