import numpy as np
import itertools
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
import torch.nn.functional as F
import torch.nn as nn

regra=2159062512564987644819455219116893945895958528152021228705752563807958532187120148734120
base1=5
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

print(cellular_automaton()[1:4,1:4])

device = torch.device("cuda")

batch_size1=500

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/home/theone/other_models/mnist', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.0,), (1.,))
                             ])),
  batch_size=batch_size1, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/home/theone/other_models/mnist', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.0,), (1,))
                             ])),
  batch_size=batch_size1, shuffle=True)


class Net(nn.Module):
    def __init__(self,kernel):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(32, 64, 3, 1,bias=False)
        self.conv2 = nn.Conv2d(1, 64, 10, 1,bias=False)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(100, 28*28)
        self.fc2 = nn.Linear(28*28, 1024)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc4 = nn.Linear(1024, 10)
        torch.nn.init.xavier_uniform(self.fc1.weight)
        torch.nn.init.xavier_uniform(self.fc2.weight)
        torch.nn.init.xavier_uniform(self.fc3.weight)
        torch.nn.init.xavier_uniform(self.fc4.weight)
        self.batch_norm = nn.BatchNorm1d(100)
        self.conv1.weight = nn.Parameter(kernel,requires_grad=False)
        self.conv2.weight = nn.Parameter(kernel,requires_grad=False)


    def forward(self, x):
        res = x.view(batch_size1, 784)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = nn.MaxPool2d(2, 2)(x)
        #x = self.conv3(x)
        #x = F.relu(x)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.batch_norm(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(torch.mean(torch.stack((x,res)),0))
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        output = F.log_softmax(x, dim=1)
        return output

import torch.optim as optim

#Adadelta 1000 epochs
n_epochs = 300
learning_rate = 0.01
log_interval = 10
train_losses = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

def norm(x):
    return (x-x.min())/(x.max()-x.min())

c=torch.from_numpy(norm(cellular_automaton()).astype(np.float16).reshape(-1,1,dimensions,dimensions)).type(torch.cuda.FloatTensor)
print(c)
net = Net(c).to(device)
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)


def train(epoch):
  net.train()
  #checkpoint = torch.load('/home/theone/other_models/Cellular Automaton/results/model_96.93.pth')
  #net.load_state_dict(checkpoint)
  for batch_idx, (data, target) in enumerate(train_loader):
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
      torch.save(net.state_dict(), '/home/theone/other_models/Cellular Automaton/results/model.pth')


def test():
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():        
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            #print(data.shape)
            try:
              output = net(data)
              test_loss += F.nll_loss(output, target, size_average=False).item()
              pred = output.max(1, keepdim=True)[1]
              correct += pred.eq(target.data.view_as(pred)).sum()
            except:
              pass
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


with torch.no_grad():
    model = Net(c)
    checkpoint = torch.load('/home/theone/other_models/Cellular Automaton/results/model.pth')
    model.load_state_dict(checkpoint)
    index = 200
    item = example_data
    image = item.to('cpu')
    true_target = example_targets[index].to('cpu')
    prediction = model.to('cpu')(image)
    predicted_class = np.argmax(prediction[index])
    image = image[index].reshape(28, 28, 1)
    plt.imshow(image, cmap='gray')
    plt.title(f'Prediction: {predicted_class} - Actual target: {true_target}')
    plt.show()

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print(params)

Train Epoch: 2 [0/60000 (0%)]   Loss: 0.025488
Train Epoch: 2 [5000/60000 (8%)]        Loss: 0.017526
Train Epoch: 2 [10000/60000 (17%)]      Loss: 0.010448
Train Epoch: 2 [15000/60000 (25%)]      Loss: 0.007660
Train Epoch: 2 [20000/60000 (33%)]      Loss: 0.009406
Train Epoch: 2 [25000/60000 (42%)]      Loss: 0.014141
Train Epoch: 2 [30000/60000 (50%)]      Loss: 0.007342
Train Epoch: 2 [35000/60000 (58%)]      Loss: 0.036374
Train Epoch: 2 [40000/60000 (67%)]      Loss: 0.020807
Train Epoch: 2 [45000/60000 (75%)]      Loss: 0.007119
Train Epoch: 2 [50000/60000 (83%)]      Loss: 0.022738
Train Epoch: 2 [55000/60000 (92%)]      Loss: 0.007261

Test set: Avg. loss: 0.0440, Accuracy: 9854/10000 (98.540001%)
