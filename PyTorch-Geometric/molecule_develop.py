import torch
from torch_geometric.data import Data
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from torch_geometric.datasets import TUDataset, Planetoid, MoleculeNet
from torch_geometric.data import DataLoader

dataset = MoleculeNet(root=".", name="ESOL")

print("Dataset features: ", dataset.num_features)
print("Dataset target: ", dataset.num_classes)
print("Dataset length: ", dataset.len)
print("Dataset sample: ", dataset[0])
print("Sample  nodes: ", dataset[0].num_nodes)
print("Sample  edges: ", dataset[0].num_edges)


print(dataset[0].x)

print(dataset[0].edge_index.t())

from rdkit import Chem
from rdkit.Chem import Draw
molecule = Chem.MolFromSmiles(dataset[0]["smiles"])

fig = Draw.MolToImage(molecule, size = (360, 360))

fig.save('/home/anaconda3/work//molecule_first.png')  


#data.num_classes

#data.num_edges

#data.num_node_features

#data.contains_isolated_nodes()

#data.contains_self_loops()

#data.is_directed()


from torch.nn import Linear
import torch
from torch.nn import Linear
import torch.nn.functional as F 
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import global_mean_pool, global_max_pool
embedding_size = 64

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        torch.manual_seed(22)

        self.initial_conv = GCNConv(dataset.num_features, embedding_size)
        self.conv1 = GCNConv(embedding_size, embedding_size)
        self.conv2 = GCNConv(embedding_size, embedding_size)
        self.out = Linear(embedding_size*2, dataset.num_classes)

    def forward(self, x, edge_index, batch_index):
        hidden = self.initial_conv(x, edge_index)
        hidden = F.tanh(hidden)
        hidden = self.conv1(hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = self.conv2(hidden, edge_index)
        hidden = F.tanh(hidden)
        hidden = torch.cat([global_max_pool(hidden, batch_index), 
                            global_mean_pool(hidden, batch_index)], dim=1)
        out = self.out(hidden)

        return out, hidden

model = Net()
print(model)
print("Parameters: ", sum(p.numel() for p in model.parameters()))

from torch_geometric.data import DataLoader

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0007)  

model = model.to("cpu")

data_size = len(dataset)
size = 32
loader = DataLoader(dataset[:int(data_size * 0.8)], 
                    batch_size=size, shuffle=True)
test_loader = DataLoader(dataset[int(data_size * 0.8):], 
                         batch_size=size, shuffle=True)

for batch in loader:
    print(batch)
    print(batch.num_graphs) 

def train(data):
    for batch in loader:
      batch.to(device)  
      optimizer.zero_grad() 
      pred, embedding = model(batch.x.float(), batch.edge_index, batch.batch) 
      loss = torch.sqrt(loss_fn(pred, batch.y))       
      loss.backward()  
      optimizer.step()   
    return loss, embedding

for epoch in range(3000):
    loss, h = train(dataset)
    if epoch % 200 == 0:
      print(f"Epoch {epoch} - Train Loss {loss}")
