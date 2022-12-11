#A Tensor library like NumPy, with strong GPU support
import torch 
#A neural networks library deeply integrated with autograd designed for maximum flexibility
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import copy
import statistics
import pandas as pd        # For loading and processing the dataset
import matplotlib.pyplot as plt

NUM_CLASSES = 6
NUM_FEATURES = 300

class MoviesDataset(Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        # Make 'genre' value numeric
        df['genre'] = df['genre'].map({'action':0,
                                    'comedy':1,
                                    'documentary':2,
                                    'drama':3,
                                    'horror':4,
                                    'thriller':5
                                    }).astype(int)

        # convert the Pandas dataframe to a NumPy array, and split it into a training and test set
        self.x = df.drop('genre', axis='columns')
        self.y = df['genre']

        

    def __getitem__(self,index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples



class Network(nn.Module):
    def __init__(self, n_nodes_list):
        super(Network, self).__init__()

        n_layers = len(n_nodes_list)

        l=[]
        for i in range(n_layers-1):
            l.append(nn.Linear(n_nodes_list[i], n_nodes_list[i+1]))

        self.layers = nn.ModuleList(l)

        # define a softmax output
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # here we define the input-output computations
        for layer in self.layers:
            x=layer(x)

        # take the last layer and normalize it into a probability distribution
        x = self.softmax(x) 
        return x


class Network_Manager():

    def get_weights(csv_file):
        # Read the CSV input file
        df = pd.read_csv(csv_file)
        df['genre'] = df['genre'].map({'action':0,
                                        'comedy':1,
                                        'documentary':2,
                                        'drama':3,
                                        'horror':4,
                                        'thriller':5
                                        }).astype(int)

        num_samples = len(df)

        weights = [0]*NUM_CLASSES

        for i in range(NUM_CLASSES):
            weights[i] = num_samples / (NUM_CLASSES*df['genre'].value_counts()[i])

        # Mean and standard deviation of the list
        stdev = statistics.pstdev(weights)
        mean = statistics.mean(weights)

        # set the mean and standard deviation of the weights vector, with a low stdev to reduce the effect of weights
        desired_stdev = 0.2
        desired_mean = 1
        weights = [ ((x-mean) / (stdev*(1/desired_stdev)))+desired_mean for x in weights]

        return torch.FloatTensor(weights)

        

    def run_epoch(is_training, optimizer, loss_function, model, dataloader):
        #is_training is a bool that tells us if we are doing training or evaluation
        loss_sum = 0    
        counter = 0

        # enumerate(dataloader) separates the data in chuncks of batch_size for us
        for k, (x_train, y_train) in enumerate(dataloader):

            if is_training:
                # zero the gradients for every batch
                optimizer.zero_grad()

            # compute the predictes output
            y_pred = model(x_train)
            
            # compute loss and its gradients
            loss = loss_function(y_pred, y_train)

            if is_training:
                loss.backward()
                # adjust the weights (do the optimitzation)
                optimizer.step()

            # Gather data
            loss_sum += loss.item()
            counter +=1

        return loss_sum/counter


    def __init__(n_nodes_list):
        """
        INPUT:
        - List that contains the number of nodes in each layer
        OUTPUT:
        - Best performing model (using validation dataset)
        - Plot loss statistics in each epoch
        """
        
        # input layer needs to have as many nodes as our feature vector
        # output layer needs to have 6 nodes to match our 6 classes
        current_model = Network(n_nodes_list)
        best_model = copy.deepcopy(current_model)
        
        # specify our optimizer: adjusts our model's adjustable parameters to fit our data (lower the loss)
        # lr is the learning rate: dictates the magnitude of changes that the optimizer can make at a time.
        optimizer = optim.Adam(current_model.parameters(), lr=0.001) 

        # epoch is a full pass through our entire data
        n_epochs = 20
        # number of samples used in one forward and backward pass
        batch = 100

        # loss function computes the difference between two probability distributions for a provided set of occurrences
        # to measure how far off the neural network is from the targeted output
        # we set weights for each class to manage imbalance in our dataset
        tr_loss_function = nn.CrossEntropyLoss(weight=get_weights("train.csv"))
        val_loss_function = nn.CrossEntropyLoss(weight=get_weights("validation.csv"))

        tr_dataset = MoviesDataset("train.csv")
        val_dataset = MoviesDataset("validation.csv")
        tr_dataloader = DataLoader(dataset=tr_dataset, batch_size=batch, shuffle=True)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch, shuffle=True)

        best_loss = 9999999 # very bad loss so that it is overwitten on the first iteration

        tr_loss_list = []
        val_loss_list = []

        #training loop
        for epoch in range(n_epochs):

            #training phase
            current_model.train(True)
            tr_l = run_epoch(True, optimizer, tr_loss_function, current_model, tr_dataloader)
            tr_loss_list.append(tr_l)

            #validation phase
            current_model.train(False)
            avg_loss = run_epoch(False, optimizer, val_loss_function, current_model, val_dataloader)
            val_loss_list.append(avg_loss)
        
            # if we obtain a better loss (on validation dataset), we save the model
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model = copy.deepcopy(current_model)


        plt.plot(tr_loss_list, label = 'training loss per epoch')
        plt.plot(val_loss_list, label = 'validation loss per epoch')
        plt.legend()
        plt.show()

        self.model = best_model  


    def get_accuracies(csv_file):
        #set testing mode
        torch.no_grad() 

        dataset = MoviesDataset(csv_file)
        dataloader = DataLoader(dataset)

        #for global accuracy
        hits = 0
        count = 0

        # get accuracy per label
        count_labels = [0] * NUM_CLASSES
        hit_labels = [0] * NUM_CLASSES

        for x, y in dataloader: 

            y_pred_not_decoded = model(x) 
            y_pred = torch.argmax(y_pred_not_decoded, dim=1)

            count += 1
            count_labels[y] += 1

            correct = (y_pred == y).item()

            hits += correct
            hit_labels[y] += correct


        # compute accuracies
        acc = hits / count

        acc_labels = [0] * NUM_CLASSES
        for i in range(NUM_CLASSES):
            acc_labels[i] = hit_labels[i] / count_labels[i]


        return acc, acc_labels


############
# IMPLEMENTATION OF THE NN
############
model = "__init__"([NUM_FEATURES, 64, NUM_CLASSES])

############
# TESTING OF THE MODEL
############
genres = ['action', 'comedy','documentary','drama','horror','thriller']

print("------------------test dataset------------------")
tst_acc, tst_acc_labels = get_accuracies("test.csv")
print('Accuracy of the model: ', 100 * tst_acc, '%') 
for i in range(NUM_CLASSES):
    print('Accuracy of the model predicting',genres[i],'genre :', 100 * tst_acc_labels[i], '%')

print("------------------train dataset------------------")
tr_acc, tr_acc_labels = get_accuracies("train.csv")
print('Accuracy of the model: ', 100 * tr_acc, '%') 
for i in range(NUM_CLASSES):
    print('Accuracy of the model predicting',genres[i],'genre :', 100 * tr_acc_labels[i], '%')

print("------------------validation dataset------------------")
val_acc, val_acc_labels = get_accuracies("validation.csv")
print('Accuracy of the model: ', 100 * val_acc, '%') 
for i in range(NUM_CLASSES):
    print('Accuracy of the model predicting',genres[i],'genre :', 100 * val_acc_labels[i], '%')

print("------------------MEAN ACCURACIES------------------")
mean_acc = (tst_acc + tr_acc + val_acc) / 3
mean_labels = [0]*NUM_CLASSES
for i in range(NUM_CLASSES):
    mean_labels[i] = (tst_acc_labels[i] + tr_acc_labels[i] + val_acc_labels[i]) / 3

print('Mean classification accuracy ', 100 * mean_acc, '%') 
for i in range(NUM_CLASSES):
    print('Mean classification accuracy of',genres[i],'genre :', 100 * mean_labels[i], '%')