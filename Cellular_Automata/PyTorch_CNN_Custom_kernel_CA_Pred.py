import numpy as np
import itertools
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim


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

device = torch.device("cuda")


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=128, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=1000, shuffle=True)

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)


import torch.nn.functional as F
import torch.nn as nn

class Net(nn.Module):
    def __init__(self,kernel):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 3,bias=False)
        self.conv2 = nn.Conv2d(1, 64, 3, 1,bias=False)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(576, 1024)
        self.fc2 = nn.Linear(1024, 10)
        self.conv1.weight = nn.Parameter(kernel,requires_grad=True)
        #self.conv2.weight = nn.Parameter(kernel,requires_grad=True)


    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

import torch.optim as optim

PATH = './cifar_net.pth'


n_epochs = 200
batch_size_train = 128
batch_size_test = 128
learning_rate = 0.008
momentum = 0.5
log_interval = 10

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

c=torch.from_numpy(cellular_automaton().astype(np.float16).reshape(-1,1,5,5)).type(torch.cuda.FloatTensor)
#print(c)
net = Net(c).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)


def train(epoch):
  net.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    #net.load_state_dict(torch.load(PATH))
    net.train()
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = net(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if batch_idx % log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
      torch.save(net.state_dict(), '/home/theone/other_models/Cellular Automaton/results/model.pth')
      torch.save(optimizer.state_dict(), '/home/theone/other_models/Cellular Automaton/results/optimizer.pth')


def test():
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            test_loss /= len(test_loader.dataset)
            test_losses.append(test_loss)
            print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))



for epoch in range(1, n_epochs + 1):
    train(epoch)
    test()

examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
# Disable grad

with torch.no_grad():
    
    # Retrieve item
    index = 256
    item = example_data[index]
    image = item[0]
    true_target = example_targets[1]
    
    # Generate prediction
    prediction = Net(image)
    
    # Predicted class value using argmax
    predicted_class = np.argmax(prediction)
    
    # Reshape image
    image = image.reshape(28, 28, 1)
    
    # Show result
    plt.imshow(image, cmap='gray')
    plt.title(f'Prediction: {predicted_class} - Actual target: {true_target}')
    plt.show()

model_parameters = filter(lambda p: p.requires_grad, Net.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)
