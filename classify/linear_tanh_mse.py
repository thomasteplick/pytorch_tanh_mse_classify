'''
Created on Nov 2, 2026
Artificial Neural Network (ANN) using linear combiners, Tanh activation function, and Mean Square Error (MSE) loss function.
Otherwise known as a multi-layer perceptron (MLP) using backpropagation of errors to train the weights and biases.
The task is to classify four categories (classes), each having ten properties uniformly distributed in regions of (0,256].
Display the learning curve and the distribution of the properties for each class.  Test the trained ANN with a separate
set of samples and tabulate the results.

The PyTorch library was used to build the ANN.

@author: Tom Teplick
'''
import torch
from torch import nn
import torch.utils.data as data
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time
import os.path
import math
from datetime import datetime

nclasses = 4
nproperties = 10
# encoded target of the nclasses, prediction has this size also
noutputs = math.ceil(math.log2(nclasses))

# Define ANN model, choose hidden layers and features in each layer
class NeuralNetwork(nn.Module):
    def __init__(self, epochs, hidden_layers, features):
        super().__init__()
        self.linear_tanh_stack = None
        match hidden_layers:
            case 1:
                self.linear_tanh_stack = nn.Sequential(
                    nn.Linear(nproperties, features),
                    nn.Tanh(),
                    nn.Linear(features, noutputs))
            case 2:           
                self.linear_tanh_stack = nn.Sequential(
                    nn.Linear(nproperties, features),
                    nn.Tanh(),
                    nn.Linear(features, features),
                    nn.Tanh(),
                    nn.Linear(features, noutputs))
            case 3:
                self.linear_tanh_stack = nn.Sequential(
                    nn.Linear(nproperties, features),
                    nn.Tanh(),
                    nn.Linear(features, features),
                    nn.Tanh(),
                    nn.Linear(features, features),
                    nn.Tanh(),
                    nn.Linear(features, noutputs))
            case _:
                print(f"{hidden_layers} is not supported: choose (1,2,3)")
        self.loss = torch.zeros(epochs, dtype=torch.float32)
        self.count = torch.zeros(nclasses, dtype=torch.int32)
        self.correct = torch.zeros(nclasses, dtype=torch.int32)
         
    def forward(self, x):
        logits = self.linear_tanh_stack(x)
        return logits
    
    def createSamples(self, nsamples, input_data, target, propbegin, propend):
        # Create the input data and the target.  The target is encoded to noutputs size
        for samp in range(nsamples):
            for prop in range(nproperties):
                cls = torch.randint(0, nclasses, (1,1), dtype=torch.uint8)
                # Encode the target to noutputs in (-1,1)
                shift = 0
                for i in range(noutputs):
                    if (cls & 1 << shift) > 0:
                        target[samp][i] = 1
                    else:
                        target[samp][i] = -1
                    shift += 1
                input_data[samp][prop] = torch.randint(propbegin[cls.item()][prop], 
                    propend[cls.item()][prop], (1,1), dtype=torch.uint8).item()
    
    def train_(self, dataloader, loss_fn, optimizer, device):
        self.train()
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
    
            # Compute prediction error
            pred = self.forward(X)
            loss = loss_fn(pred, y.squeeze(1))
    
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # save the loss for learning curve display
            self.loss[batch] = loss.item()
            
    def test_(self, dataloader, device):
        self.eval()
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = self.forward(X)
                # loop over y tensor and determine if pred is equal,
                # count the number of samples for each class
                for i, target in enumerate(y):
                    # decode y and pred to the integer class
                    pwr = 1
                    # true class
                    cls = 0
                    # predicted class
                    predcls = 0
                    for j in range(noutputs):
                        if target[j].item() > 0:
                            cls += pwr
                        # zero is the threshold for -1 and 1 decision
                        if pred[i][j] > 0:
                            predcls += pwr
                        pwr *= 2
                        #print(f"{target[j].item()},{pred[i][j]} ", end='')
                    self.count[cls] += 1
                    if cls == predcls:
                        self.correct[cls] += 1
                    #print()    
    def displayTestResults(self):
               
        # show classes, count, correct, %correct, totals
        # heading
        ttlCorrect = 0
        ttlCount = 0
        sep = 8
        horzBorder = '|' + '-'*sep + '|' + '-'*sep + '|' + '-'*sep + '|' + '-'*sep + '|'
        print(horzBorder)
        print("|{0:8s}|{1:8s}|{2:8s}|{3:8s}|".format("Class", "Count", "Correct", "%Correct"))
        print(horzBorder)
        for i, val in enumerate(self.count):
            print("|{0:<8d}|{1:<8d}|{2:<8d}|{3:<8d}|".format(i+1, val, self.correct[i], self.correct[i]*100//val))
            print(horzBorder)
            ttlCount += val
            ttlCorrect += self.correct[i]
        print("|{0:<8s}|{1:<8d}|{2:<8d}|{3:<8d}|".format("Totals", ttlCount, ttlCorrect, round((ttlCorrect/ttlCount*100).item())))
        print(horzBorder)
        
    def  plotClassDistribution(self, propbegin, propend, maxProp):
        # plot the ranges of the class properties which are regions in (0, 255)
        # x-axis is a unitless number running from 0 to 10*prop-space + 4*(class-space+class-width) + property-space
        # y-axis is an integer running from 0 to 255
        # loop over 10 properties
        #   skip property space
        #   loop over 4 property ranges with space in-between each property range
        # set x-label, y-label, title, xticks, classSpace, propSpace, legend, line colors, line width
        # show figure
        
        # x-axis spacing between items: classes and properties
        classSpace = 2
        propSpace = nclasses*classSpace
        # line colors
        clr = np.array(['red', 'green', 'blue', 'orange'])
        # line properties: width, style
        lnwid = 3
        lnsty = "solid"
        lbl = np.array(['class1', 'class2', 'class3', 'class4'])
        # x,y coordinates arrays to be scaled and sliced
        xcoord = np.ones(maxProp, dtype=np.uint8)
        ycoord = np.arange(maxProp, dtype=np.uint8)
        xtick = np.zeros(nproperties, dtype=np.uint8)
                
        x = 0
        _, ax = plt.subplots(1,1)
        # loop over the properties
        for prop in range(nproperties):
            x += propSpace
            # loop over the classes
            for cls in range(nclasses):
                # property y length
                ln = propend[cls][prop] - propbegin[cls][prop]
                if prop == 0:
                    ax.plot(x*xcoord[:ln], ycoord[propbegin[cls][prop]:propend[cls][prop]], label=lbl[cls],
                            color=clr[cls], linewidth=lnwid, linestyle=lnsty)
                else:
                    ax.plot(x*xcoord[:ln], ycoord[propbegin[cls][prop]:propend[cls][prop]],
                            color=clr[cls], linewidth=lnwid, linestyle=lnsty)                    
                # put the property value at this x coordinate
                if cls == nclasses//2:
                    xtick[prop] = x
                x += classSpace
        ax.legend()
        ax.set_xlabel('Class Properties')
        ax.set_ylabel('Property Values (raw)')
        ax.set_title('Class Property Distribution')
        # custom x-ticks for properties
        ax.set_xticks(xtick, np.arange(nproperties)+1)
        plt.show(block=True)  
                
    def plotLearningCurve(self):
        # show Mean Square Error loss per epoch from training the MLP
        _, ax = plt.subplots(1,1)
        x = np.arange(self.loss.size(0), dtype=np.uint32)
        y = self.loss.numpy()
        # line properties: width, style
        lnwid = 3
        lnsty = "solid"
        ax.plot(x, y, linewidth=lnwid, linestyle=lnsty)        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean Square Error Loss')
        ax.set_title('Learning Curve')
        plt.show(block=True)          
                                          
def runLinearTanhMSE():
    
    parser = argparse.ArgumentParser(prog='linear_tanh_mse', description="MLP neural network (Tanh, Mean Square Error Loss) for classification using PyTorch")
    parser.add_argument('epochs', type=int, choices=[100, 200, 500, 1000, 2000, 5000, 10000], help="the number of epochs for training and testing")
    parser.add_argument('batch_size', type=int, choices=[1, 2, 10, 16, 32, 64], help="the batch size for training and testing")
    parser.add_argument('learning_rate', type=float, choices=[.001, .002, .01, .02], help="learning rate hyperparameter for training")
    parser.add_argument('momentum', type=float, choices=[.5, .6, .7, .8, .9], help="momentum hyperparameter for training")
    parser.add_argument('hidden_layers', type=int, choices=[1,2,3], help="number of hidden layers, depth of MLP")
    parser.add_argument('features', type=int, choices=[10,15,20], help="hidden layer width")
    args = parser.parse_args()
    
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")
    
    # seed the random number generator
    torch.manual_seed(int(time.time()))
    
    # each epoch consists of batch_size samples which consists of (input,target)
    epochs = args.epochs  #args.epochs
    # number of samples consisting of (input,target)
    batch_size = args.batch_size #args.batch_size
    # learning rate to scale the backpropagation error
    learning_rate = args.learning_rate #args.learning_rate
    # momentum
    momentum = args.momentum    
    # hidden layers of the MLP
    hidden_layers = args.hidden_layers   
    # features of the layers or width
    features = args.features
    
    # min and max of property deltas
    lodelta = 25
    hidelta = 50

    maxProp = 256
    
    nsamples = epochs*batch_size
    
    linearTanhMSEModel = "data\\linearTanhMSE.pth"
    linearTanhMSEData = "data\\linearTanhMSE.txt"
   
    propbegin = torch.empty((nclasses, nproperties), dtype=torch.uint8)
    propend = torch.empty((nclasses, nproperties), dtype=torch.uint8)
    # Set class property bounds if data file doesn't exist, otherwise read in the file
    if os.path.exists(linearTanhMSEData):
        with open(linearTanhMSEData, mode='r', encoding="utf-8") as fin:
            cls = 0
            for line in fin:
                bounds = line.split(' ')
                for prop in range(nproperties):
                    propbegin[cls][prop] = int(bounds[2*prop])
                    propend[cls][prop] = int(bounds[2*prop+1])
                cls += 1
    else:        
        # split the properties in half, assign first half starting in (0, maxProperty/2), second half ending in (maxProperty/2, maxProperty) 
        
        # first half, start in (0, maxProperty/2)
        propbegin[:, 0:nproperties//2] = torch.randint(0, maxProp//2, (nclasses, nproperties//2), dtype=torch.uint8)
        propend[:, 0:nproperties//2] = propbegin[:, 0:nproperties//2]  + \
            torch.randint(lodelta, hidelta, (nclasses, nproperties//2), dtype=torch.uint8)
        
        # second half, end in (maxProperty/2, maxProperty)
        propend[:, nproperties//2:nproperties] = torch.randint(maxProp//2, maxProp, (nclasses, nproperties//2), dtype=torch.uint8)
        propbegin[:, nproperties//2:nproperties] = propend[:, nproperties//2:nproperties] - \
            torch.randint(lodelta, hidelta, (nclasses, nproperties//2), dtype=torch.uint8)
            
        # shuffle the properties of each class, create a permutation of arange(nproperties)
        perm = torch.randperm(nproperties)
        propbegin = propbegin[:, perm]
        propend = propend[:, perm]
    
    # Input data and the target tensors comprise the samples used to train the ANN
    input_data = torch.empty(nsamples, nproperties, dtype=torch.uint8)
    target = torch.empty(nsamples, noutputs, dtype=torch.int8)
    
    # create the ANN
    model = NeuralNetwork(epochs, hidden_layers, features).to(device)
    print(f"Model: {model}")    
    
    # Train the model
    
    # create samples from the begin/end properties            
    model.createSamples(nsamples, input_data, target, propbegin, propend)
        
    # Create dataset from sample tensors, normalize the input data to (-1,1)
    dataset = data.TensorDataset(2.0*(input_data/maxProp-0.5), target)
    # Create dataloader from dataset
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Display the class distributions using matplotlib
    print("Display the class distributions")
    model.plotClassDistribution(propbegin, propend, maxProp)
       
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    trainbegin = datetime.now()   
    print(f"Begin training the model: {trainbegin}")
    for t in range(epochs):
        print("*", end='', flush=True)
        model.train_(dataloader, loss_fn, optimizer, device)
        if (t+1) % 50 == 0:
            print()
    trainend = datetime.now()
    elapsedtime = trainend-trainbegin
    print(f"\nDone training: {trainend}")
    print(f"Elapsed time: {elapsedtime}")

    # Display MSE Loss using matplotlib
    print("Display the Mean Square Error Loss")
    model.plotLearningCurve()
    
    torch.save(model.state_dict(), linearTanhMSEModel)
    print("Saved PyTorch Model State to %s" % linearTanhMSEModel)
    
    # Save the class properties
    with open(linearTanhMSEData, mode='w', encoding="utf-8") as fout:
        for cls in range(nclasses):
            for prop in range(nproperties):
                print(f"{propbegin[cls][prop]} {propend[cls][prop]} ", file=fout, end='')
            print(file=fout)
            
    # Test the model using different samples
    
    # create samples from the begin/end properties            
    model.createSamples(nsamples, input_data, target, propbegin, propend)
                
    # Create dataset from sample tensors, normalize the input data to (-1,1)
    dataset = data.TensorDataset(2.0*(input_data/maxProp-0.5), target)
    # Create dataloader from dataset
    dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    testbegin = datetime.now()   
    print(f"Begin testing the model: {testbegin}")
    model.test_(dataloader, device)
    testend = datetime.now()
    elapsedtime = testend-testbegin
    print(f"\nDone testing.  {testend}")
    print(f"Elapsed time: {elapsedtime} Test Results:")
        
    # Tabulate and display the test results: class, count, correct, %correct, totals
    model.displayTestResults()
    
if __name__ == '__main__':
    runLinearTanhMSE()