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

class Net(nn.Module):
    def __init__(self,kernel):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3,bias=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3,bias=False)
        self.fc1 = nn.Linear(800, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 100)
        self.conv1.weight = nn.Parameter(kernel,requires_grad=True)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))).view(5,6,-1,16)
        #print(x.shape)
        x = self.pool(F.relu(self.conv2(x).view(10,1,16,21)))
        x = x.view(-1, 800)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        #print('output',x.shape)
        return F.log_softmax(x,dim=1).view(-1,10)

import torch.optim as optim

PATH = './cifar_net.pth'

c=torch.from_numpy(cellular_automaton()[1:4,1:4].astype(np.float16).reshape(1,3,3,1)).type(torch.FloatTensor)
#print(c)
net = Net(c)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
#net.load_state_dict(torch.load(PATH))
loss = nn.CrossEntropyLoss()
input = torch.randn(3, 5, requires_grad=True)
target = torch.empty(3, dtype=torch.long).random_(5)
output = loss(input, target)
output.backward()






for epoch in range(2):  # loop over the dataset multiple times


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
        ##print(outputs)
        #print(labels)
        loss = criterion(outputs, labels)
        #print(loss)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
    torch.save(net.state_dict(), PATH)



correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
