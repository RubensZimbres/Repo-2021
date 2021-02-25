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
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
trainTransform  = tv.transforms.Compose([tv.transforms.ToTensor(), tv.transforms.Normalize((0.1307,), (0.3081,))])
trainset = tv.datasets.MNIST(root='./data',  train=True,download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=False, num_workers=4)
testset = tv.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)


import torch.nn.functional as F
import torch.nn as nn


class Net(nn.Module):
    def __init__(self,kernel):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 1, kernel_size=5,bias=False)
        self.conv2 = nn.Conv2d(1, 3, kernel_size=5)
        self.fc1 = nn.ConvTranspose2d(3, 1, kernel_size=5,bias=False)
        self.fc2 = nn.ConvTranspose2d(1, 1, kernel_size=5,bias=False)
        self.up=nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv1.weight = nn.Parameter(kernel,requires_grad=False)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv2(self.conv1(x.view(32,1,28,28))), 2))
        x = F.relu(self.up(x))
        x = F.relu(self.fc1(x))
        x = F.relu(F.max_pool2d(self.fc2(x),2))
        return F.relu(x)

import torch.optim as optim

PATH = './cifar_net.pth'


n_epochs = 3
batch_size_train = 128
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(dataloader.dataset) for i in range(n_epochs + 1)]


def train(epoch):
  for batch_idx, (data, target) in enumerate(dataloader):
    #net.load_state_dict(torch.load(PATH))
    c=torch.from_numpy(cellular_automaton().astype(np.float16).reshape(1,1,5,5)).type(torch.FloatTensor)
    #print(c)
    net = Net(c)
    criterion = nn.KLDivLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

    net.train()
    optimizer.zero_grad()
    output = net(data)
    #print(output.view(32,196).shape)
    #print(F.interpolate(data.view(32,1,784), size=196).view(32,196).shape)
    loss = criterion(output.view(32,196),F.interpolate(data.view(32,1,784), size=196).view(32,196))
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(dataloader.dataset),
        100. * batch_idx / len(dataloader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(dataloader.dataset)))
      torch.save(net.state_dict(), './results/model.pth')
      torch.save(optimizer.state_dict(), './results/optimizer.pth')


def test():
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, data in testloader:
            output = net(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            test_loss /= len(testloader.dataset)
            test_losses.append(test_loss)
            print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(testloader.dataset),
                100. * correct / len(testloader.dataset)))



for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()
