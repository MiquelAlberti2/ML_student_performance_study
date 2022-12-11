import copy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

matplotlib.use('TkAgg')

df1 = pd.read_csv("dataset2/student-mat.csv", delimiter=";")
df2 = pd.read_csv("dataset2/student-por.csv", delimiter=";")

df = pd.concat([df1, df2])

col_to_del = ['Mjob', 'Fjob', 'school', 'reason', 'guardian', 'schoolsup', 'famsup', 'nursery', 'Dalc', 'Walc', 'G1',
              'G2']

df.drop(columns=col_to_del, inplace=True, axis=1)

df['sex'] = df['sex'].map({'F': 0,
                           'M': 1,
                           }).astype(int)

df['address'] = df['address'].map({'U': 0,
                                   'R': 1,
                                   }).astype(int)

df['famsize'] = df['famsize'].map({'LE3': 0,  # less or equal than 3
                                   'GT3': 1,  # greater than 3
                                   }).astype(int)

df['Pstatus'] = df['Pstatus'].map({'T': 0,
                                   'A': 1,
                                   }).astype(int)

"""df['Mjob'] = df['Mjob'].map({'teacher':1,
                             'health':2,
                             'services':3,
                             'at_home':4,
                             'other':5
                            }).astype(int)

df['Fjob'] = df['Fjob'].map({'teacher':1,
                             'health':2,
                             'services':3,
                             'at_home':4,
                             'other':5
                            }).astype(int)"""

df['higher'] = df['higher'].map({'no': 0,
                                 'yes': 1,
                                 }).astype(int)

df['internet'] = df['internet'].map({'no': 0,
                                     'yes': 1,
                                     }).astype(int)

df['paid'] = df['paid'].map({'no': 0,
                             'yes': 1,
                             }).astype(int)

df['activities'] = df['activities'].map({'no': 0,
                                         'yes': 1,
                                         }).astype(int)

df['romantic'] = df['romantic'].map({'no': 0,
                                     'yes': 1,
                                     }).astype(int)

train, test = train_test_split(df, test_size=0.2)

train_numpy = train.to_numpy()
test_numpy = test.to_numpy()

X_train = train_numpy[:, :-1]
y_train = train_numpy[:, -1]
X_test = test_numpy[:, :-1]
y_test = test_numpy[:, -1]

# Train Data
train_data = pd.read_csv("data/train.csv")
# Make 'genre' value numeric
train_data["genre"] = train_data['genre'].map({'action': 0,
                                               'comedy': 1,
                                               'documentary': 2,
                                               'drama': 3,
                                               'horror': 4,
                                               'thriller': 5}).astype(int)
# To get data
"""X_train = train_data.drop(["genre"], axis=1).values
y_train = train_data["genre"].values"""
# Scaling The Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Getting tensor
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)

"""# Validation Data
val_data = pd.read_csv("data/validation.csv")
val_data["genre"] = val_data['genre'].map({'action': 0,
                                           'comedy': 1,
                                           'documentary': 2,
                                           'drama': 3,
                                           'horror': 4,
                                           'thriller': 5}).astype(int)
print(val_data)
# To get data
X_val = val_data.drop(["genre"], axis=1).values
y_val = val_data["genre"].values
# Scaling The Data
scaler = StandardScaler()
X_val = scaler.fit_transform(X_val)
# Getting tensor
X_val = torch.FloatTensor(X_val)
y_val = torch.LongTensor(y_val)"""

"""# Test data
test_data = pd.read_csv("data/test.csv")
test_data["genre"] = test_data['genre'].map({'action': 0,
                                             'comedy': 1,
                                             'documentary': 2,
                                             'drama': 3,
                                             'horror': 4,
                                             'thriller': 5}).astype(int)
# To get data
X_test = test_data.drop(["genre"], axis=1).values
y_test = test_data["genre"].values"""
# Scaling The Data
scaler = StandardScaler()
X_test = scaler.fit_transform(X_test)
# Getting tensor
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)


class NeuralNetworkClassificationModel(nn.Module):
    def __init__(self, input_dim, num_hidden_layers, hidden_size, output_dim):
        super(NeuralNetworkClassificationModel, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_size)

        # Create remaining hidden layers
        self.hidden_layers = nn.ModuleList()
        for i in range(0, num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))

        self.output_layer = nn.Linear(hidden_size, output_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            out = self.sigmoid(layer(out))
            out = self.dropout(out)
        out = self.output_layer(out)
        return out


input_dim = 300
output_dim = 1
num_hidden_layers = 1
hidden_size = 64
model = NeuralNetworkClassificationModel(input_dim, num_hidden_layers, hidden_size, output_dim)

# creating our optimizer and loss function object
learning_rate = 0.01
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


def train_network(model, optimizer, criterion, X_train, y_train, X_test, y_test, num_epochs, train_losses, test_losses):
    best_loss = 1000
    for epoch in range(num_epochs):
        # clear out the gradients from the last step loss.backward()
        optimizer.zero_grad()

        # forward feed
        model.train(True)
        output_train = model(X_train)

        # calculate the loss
        loss_train = criterion(output_train, y_train)

        # backward propagation: calculate gradients
        loss_train.backward()

        # update the weights
        optimizer.step()

        # validation phase
        # model.train(False)
        # output_val = model(X_val)
        # loss_val = criterion(output_val, y_val)

        """if loss_val < best_loss:
            best_loss = loss_val
            best_model = copy.deepcopy(model)"""

        output_test = model(X_test)
        loss_test = criterion(output_test, y_test)

        train_losses[epoch] = loss_train.item()
        test_losses[epoch] = loss_test.item()
        # valid_losses[epoch] = loss_val.item()

        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {loss_train.item():.4f}")
            # print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {loss_train.item():.4f}, "
                  #f"Validation Loss: {loss_val.item():.4f}")


num_epochs = 1000
train_losses = np.zeros(num_epochs)
test_losses = np.zeros(num_epochs)

train_network(model, optimizer, criterion, X_train, y_train, X_test, y_test, num_epochs, train_losses, test_losses)

# Plot the results
plt.figure(figsize=(10, 10))
plt.plot(train_losses, label='Train Loss')
# plt.plot(valid_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

# Get the predictions
with torch.no_grad():
    predictions_train = model(X_train)
    predictions_test = model(X_test)
    # predictions_validation = model(X_val)


# Check how the predicted outputs look like and after taking argmax compare with y_train or y_test
# predictions_train
# y_train,y_test

def get_accuracy_multiclass(pred_arr, original_arr):
    if len(pred_arr) != len(original_arr):
        return False
    pred_arr = pred_arr.numpy()
    original_arr = original_arr.numpy()
    final_pred = []
    # we will get something like this in the pred_arr [32.1680,12.9350,-58.4877]
    # so will be taking the index of that argument which has the highest value here 32.1680 which corresponds
    # to 0th index
    for i in range(len(pred_arr)):
        final_pred.append(np.argmax(pred_arr[i]))
    final_pred = np.array(final_pred)
    count = 0
    # 'action': 0,'comedy': 1,'documentary': 2,'drama': 3,'horror': 4,'thriller': 5
    countperclass = {'0': 0,
                     '1': 0,
                     '2': 0,
                     '3': 0,
                     '4': 0,
                     '5': 0}
    # Comparison between the predicted_arr and the original_arr to get the mean final accuracy
    for j in range(len(original_arr)):
        if final_pred[j] == original_arr[j]:
            count += 1
            countperclass[str(final_pred[j])] += 1
    meanperclass = {'action': countperclass['0'] / np.count_nonzero(final_pred == 0),
                    'comedy': countperclass['1'] / np.count_nonzero(final_pred == 1),
                    'documentary': countperclass['2'] / np.count_nonzero(final_pred == 2),
                    'drama': countperclass['3'] / np.count_nonzero(final_pred == 3),
                    'horror': countperclass['4'] / np.count_nonzero(final_pred == 4),
                    'thriller': countperclass['5'] / np.count_nonzero(final_pred == 5)}
    return (count / len(final_pred)), meanperclass


train_acc, train_class_acc = get_accuracy_multiclass(predictions_train, y_train)
test_acc, test_class_acc = get_accuracy_multiclass(predictions_test, y_test)
# val_acc, val_class_acc = get_accuracy_multiclass(predictions_validation, y_val)

print('------- ACCURACIES -------')
print('---Training---')
print(f"Accuracy: {round(train_acc * 100, 3)}")
print('Mean Per Class')
for i in train_class_acc:
    print(f"\tTraining Class '{i}' Mean Accuracy: {round(train_class_acc[i] * 100, 3)}")

print()
print('---Test---')
print(f"Accuracy: {round(test_acc * 100, 3)}")
print('Mean Per Class')
for i in test_class_acc:
    print(f"\tTest Class '{i}' Mean Accuracy: {round(test_class_acc[i] * 100, 3)}")

"""print()
print('---Validation---')
# print(f"Accuracy: {round(val_acc * 100, 3)}")
print('Mean Per Class')
for i in test_class_acc:
    print(f"\tTest Class '{i}' Mean Accuracy: {round(val_class_acc[i] * 100, 3)}")"""
