import numpy as np
import itertools
regra=30 #2159062512564987644819455219116893945895958528152021228705752563807958532187120148734120
base1=2
states=np.arange(0,base1)
dimensions=5
kernel=np.random.randint(len(states), size=(dimensions,dimensions))


def cellular_automaton():
    global kernel

    lista=states
    kernel=np.pad(kernel, (1, 1), 'constant', constant_values=(0))
    q12=np.array([p for p in itertools.product(lista, repeat=3)])[::-1]

    uau12 = np.zeros(q12.shape[0])
    temp = [int(i) for i in np.base_repr(int(regra),base=base1)]
    uau12[-len(temp):]=temp
    ru12=np.array(range(0,len(uau12)))

    tod12=[]
    for i in range(0,len(uau12)):
        tod12.append([0,int(uau12[i]),0])

    final=[]
    for i in range(0,len(q12)):
        final.append(np.array([q12[i],np.array(tod12).astype(np.int8)[i]]))

    
    def ca(row):
        out=[]
        for xx in range(0,dimensions):
            out.append(tod12[next((i for i, val in enumerate(q12) if np.all(val == kernel[row][xx:xx+3])), -1)][1])
        return out
    kernel=np.array([item for item in map(ca,range(1,kernel.shape[0]-1))])
    return kernel

cellular_automaton()[1:4,1:4]

import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=10,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
dataiter = iter(trainloader)
images, labels = dataiter.next()

input_size = 784
hidden_sizes = [128, 64]
output_size = 10

import torch.nn.functional as F
import torch.nn as nn

class CustomConv(nn.Module):
    def __init__(self, kernel):
        super(CustomConv, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, 3, 1, 1, bias=False)
        # Add other layers here
        
        # Initialize conv1 with custom kernel
        self.conv1.weight = nn.Parameter(kernel)
        
    def forward(self, x):
        x = self.conv1(x)
        # pass x to other modules
        return x        

model = CustomConv(torch.from_numpy(cellular_automaton()[1:4,1:4].reshape(1,3,3,1)).type(torch.FloatTensor))
x = torch.randn(1, 3, 3, 1)
output = model(x)

class Net(nn.Module):
    def __init__(self,kernel):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, kernel_size=3,bias=False)
        self.conv2 = nn.Conv2d(1, 20, kernel_size=3,bias=False)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(336, 10)
        self.fc2 = nn.Linear(25, 10)
        self.conv1.weight = nn.Parameter(kernel)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 336)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x.view(10,-1))
        return F.log_softmax(x)


import torch.optim as optim


for epoch in range(2):  # loop over the dataset multiple times

    c=torch.from_numpy(cellular_automaton()[1:4,1:4].astype(np.float16).reshape(1,3,3,1)).type(torch.FloatTensor)
    print(c)
    net = Net(c)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        #print(inputs.shape)
        #print(labels.shape)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        #print(outputs.shape)
        loss = criterion(outputs, labels)
        print(loss)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

net = Net()
net.load_state_dict(torch.load(PATH))

outputs = net(images)
