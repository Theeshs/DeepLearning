# importing libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable

# importing the data sets
movies = pd.read_csv('./P16-AutoEncoders/AutoEncoders/ml-1m/ml-1m/movies.dat', sep="::", header=None, engine="python",
                     encoding="latin-1")
users = pd.read_csv('./P16-AutoEncoders/AutoEncoders/ml-1m/ml-1m/users.dat', sep="::", header=None, engine="python",
                    encoding="latin-1")

ratings = pd.read_csv('./P16-AutoEncoders/AutoEncoders/ml-1m/ml-1m/ratings.dat', sep="::", header=None, engine="python",
                      encoding="latin-1")

# training set
training_set = pd.read_csv('./P16-AutoEncoders/AutoEncoders/ml-100k/ml-100k/u1.base', delimiter="\t")
training_set = np.array(training_set, dtype='int')

# test_set
test_set = pd.read_csv('./P16-AutoEncoders/AutoEncoders/ml-100k/ml-100k/u1.test', delimiter="\t")
test_set = np.array(test_set, dtype='int')

# creating the number of users and movies
nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))


# converting the data  into an array with users in lines and movies in columns
def convert(data):
    new_data = []
    for id_user in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_user]
        id_ratings = data[:, 2][data[:, 0] == id_user]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data


training_set = convert(training_set)
test_set = convert(test_set)

# print(test_set)
# exit()

# convert data into torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)


class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_movies, 1000)
        self.fc2 = nn.Linear(1000, 800)
        self.fc3 = nn.Linear(800, 600)
        self.fc4 = nn.Linear(600, 400)
        self.fc5 = nn.Linear(400, 200)
        self.fc6 = nn.Linear(200, 100)
        self.fc7 = nn.Linear(100, 50)
        self.fc8 = nn.Linear(50, 100)
        self.fc9 = nn.Linear(100, 200)
        self.fc10 = nn.Linear(200, 400)
        self.fc11 = nn.Linear(400, 600)
        self.fc12 = nn.Linear(600, 800)
        self.fc13 = nn.Linear(800, 1000)
        self.fc14 = nn.Linear(1000, nb_movies)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.activation(self.fc5(x))
        x = self.activation(self.fc6(x))
        x = self.activation(self.fc7(x))
        x = self.activation(self.fc8(x))
        x = self.activation(self.fc9(x))
        x = self.activation(self.fc10(x))
        x = self.activation(self.fc11(x))
        x = self.activation(self.fc12(x))
        x = self.activation(self.fc13(x))
        x = self.fc14(x)
        return x


sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr=1e-4, weight_decay=0.4)

# Training the SAE
nb_epoch = 200
for epoch in range(1, nb_epoch + 1):
    train_loss = 0.
    s = 0.
    for id_user in range(nb_users):
        input = Variable(training_set[id_user]).unsqueeze(0)
        target = input.clone()
        if torch.sum(target.data > 0) > 0:
            output = sae(input)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = nb_movies / float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.data * mean_corrector)
            s += 1
            optimizer.step()
    print('epoch: ' + str(epoch) + ' loss: ' + str(train_loss / s))

# testign SAE
test_loss = 0.
s = 0.
for id_user in range(nb_users):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = Variable(test_set[id_user])
    if torch.sum(target.data > 0) > 0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nb_movies / float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data * mean_corrector)
        s += 1.
print('test loss: ' + str(test_loss / s))
