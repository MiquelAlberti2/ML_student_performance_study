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



class MoviesDataset(Dataset):
    def __init__(self, X, y):
        # convert data into pytorch tensors
        self.x = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).type(torch.LongTensor)

        self.n_samples = self.x.shape[0]

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
            l.append(nn.ReLU())

        self.layers = nn.ModuleList(l)


    def forward(self, x):
        # here we define the input-output computations
        for layer in self.layers:
            x=layer(x)

        return x

    
class My_Network():

    def __init__(self,n_nodes_list, x_tr, x_val, y_tr, y_val):
        """
        INPUT:
        - List that contains the number of nodes in each layer
        OUTPUT:
        - Best performing model (using validation dataset)
        - Plot loss statistics in each epoch
        """
        
        # input layer needs to have as many nodes as our feature vector
        # output layer needs to have 6 nodes to match our 6 classes
        model = Network(n_nodes_list)
        best_model = copy.deepcopy(model)
        
        # specify our optimizer: adjusts our model's adjustable parameters to fit our data (lower the loss)
        # lr is the learning rate: dictates the magnitude of changes that the optimizer can make at a time.
        optimizer = optim.Adam(model.parameters(), lr=0.001) 

        # epoch is a full pass through our entire data
        n_epochs = 100
        # number of samples used in one forward and backward pass
        batch = 32

        # loss function measures how far off the neural network is from the targeted output
        loss_function = nn.L1Loss()
        #loss_function = nn.MSELoss()

        tr_dataset = MoviesDataset(x_tr, y_tr)
        val_dataset = MoviesDataset(x_val, y_val)
        tr_dataloader = DataLoader(dataset=tr_dataset, batch_size=batch, shuffle=True)
        val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch, shuffle=True)

        best_loss = 9999999 # very bad loss so that it is overwitten on the first iteration

        tr_loss_list = []
        val_loss_list = []

        #training loop
        for epoch in range(n_epochs):

            #training phase
            model.train(True)
            tr_l = self.run_epoch(True, optimizer, loss_function, model, tr_dataloader)
            tr_loss_list.append(tr_l)

            #validation phase
            model.train(False)
            avg_loss = self.run_epoch(False, optimizer, loss_function, model, val_dataloader)
            val_loss_list.append(avg_loss)
        
            # if we obtain a better loss (on validation dataset), we save the model
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model = copy.deepcopy(model)


        plt.plot(tr_loss_list, label = 'Training Loss per Epoch')
        plt.plot(val_loss_list, label = 'Validation Loss per Epoch')
        plt.legend()
        plt.show()

        self.best_model = best_model 


    def run_epoch(self,is_training, optimizer, loss_function, model, dataloader):
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
            y_train = y_train.unsqueeze(1)
            loss = loss_function(y_pred, y_train.float())

            if is_training:
                loss.backward()
                # adjust the weights (do the optimitzation)
                optimizer.step()

            # Gather data
            loss_sum += loss.item()
            counter +=1

        return loss_sum/counter


    def get_accuracy(self,x_data, y_data):
        #set testing mode
        torch.no_grad() 

        dataset = MoviesDataset(x_data, y_data)
        dataloader = DataLoader(dataset)

        #for global accuracy
        total_error = 0
        count = 0

        for x, y in dataloader: 

            y_pred = self.best_model(x) 
            error = abs(y_pred.item() - y.item())
            total_error += error
            count += 1

        # compute accuracies
        acc = total_error / count

        return acc

