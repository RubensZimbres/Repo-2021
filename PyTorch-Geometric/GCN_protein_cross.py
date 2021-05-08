from torch_geometric.datasets import TUDataset, ModelNet, ShapeNet
import networkx as nx
from torch_geometric import utils
from torch_geometric.nn import GCNConv, JumpingKnowledge, global_add_pool
from torch.nn import functional as F
import torch
from torch_geometric.data import Batch
from torch.nn import Linear
import torch
from torch.nn import Linear
import torch.nn.functional as F 
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import global_mean_pool, global_max_pool

dataset = TUDataset(root='./data/', name='PROTEINS')

G = utils.to_networkx(dataset[0])
nx.draw_kamada_kawai(G)

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

device="cpu"
model = Net()
print(model)
print("Parameters: ", sum(p.numel() for p in model.parameters()))

from torch_geometric.data import DataLoader

loss_fn = torch.nn.CrossEntropyLoss()
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
    batch.y

def train(data):
    for batch in loader:
      batch.to(device)  
      optimizer.zero_grad() 
      pred, embedding = model(batch.x.float(), batch.edge_index, batch.batch) 
      loss = loss_fn(pred, batch.y.long())
      loss.backward()  
      optimizer.step()   
    return loss, embedding

for epoch in range(2000):
    loss, h = train(dataset)
    if epoch % 100 == 0:
      print(f"Epoch {epoch} - Train Loss {loss}")
