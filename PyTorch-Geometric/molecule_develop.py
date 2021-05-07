import torch
from torch_geometric.data import Data
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from torch_geometric.datasets import TUDataset, Planetoid, MoleculeNet
from torch_geometric.data import DataLoader

dataset = MoleculeNet(root=".", name="ESOL")

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, test_size])
print(train_dataset, len(train_dataset))
print(test_dataset, len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

next(iter(test_loader))

print("Dataset features: ", dataset.num_features)
print("Dataset target: ", dataset.num_classes)
print("Dataset length: ", dataset.len)
print("Dataset sample: ", dataset[0])
print("Sample  nodes: ", dataset[0].num_nodes)
print("Sample  edges: ", dataset[0].num_edges)

for batch in loader:
    print(batch)
    print(batch.num_graphs) 

print(dataset[0].x)

print(dataset[0].edge_index.t())

from rdkit import Chem
from rdkit.Chem import Draw
molecule = Chem.MolFromSmiles(dataset[0]["smiles"])

fig = Draw.MolToImage(molecule, size = (120, 120))

fig.save('/home/anaconda3/work/home/anaconda3/work/molecule_first.png')   
#data.num_classes

#data.num_edges

#data.num_node_features

#data.contains_isolated_nodes()

#data.contains_self_loops()

#data.is_directed()

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

import torch
from torch.nn import Linear
import torch.nn.functional as F 

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)
    def forward(self, data):
        x, edge_index = data.x.float(), data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.relu(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)

for param in model.parameters():
    param.requires_grad = True

data = dataset[0].to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

model.train()
criterion=torch.nn.MSELoss()

for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(torch.tensor([torch.mean(out,axis=0)]), data.y)
    print(loss)
    loss.requires_grad = True
    loss.backward()
    optimizer.step()

