#nltk.download('averaged_perceptron_tagger')
#!pip3 install spacy graph-nets tensorflow_probability pyqt5==5.12 pyqtwebengine==5.12 gast==0.3.3
#! pip3 install dgl tensorflow transformers
#!python -m spacy download en_core_web_sm
#!pip install graph-nets
#!pip install tensorflow_probability
#!pip install pyqt5==5.12
#! pip install pyqtwebengine==5.12
#!pip install gast==0.3.3

import pandas as pd
import numpy as np
import nltk
from nltk import word_tokenize
import spacy
from spacy import displacy
from nltk import Tree
#nltk.download('punkt')

en_nlp = spacy.load('en_core_web_sm')
phrase="The quick brown fox jumps over the lazy dog"
doc = en_nlp(phrase)
root=[i.root for i in doc.sents]

words=word_tokenize(phrase)

for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
            token.shape_, token.is_alpha, token.is_stop)

select=[]
mask_init=[]
for token in doc:
    if token.pos_=='PROPN' or token.pos_=='NOUN':
        print(token.text,token.pos_)
        mask_init.append(token.i)
        select.append(token.text)

#### ESTRUTURA

def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return [node.orth_, [to_nltk_tree(child) for child in node.children]]
    else:
        return [node.orth_]

import re
import networkx as nx
import matplotlib.pyplot as plt

structure=[to_nltk_tree(sent.root) for sent in doc.sents]

def recurse(l, parent=None):
    assert isinstance(l, list)
    for item in l:
        if isinstance(item, str):
            if parent is not None:
                yield (parent, item)
            parent = item
        elif isinstance(item, list):
            yield from recurse(item, parent)
        else:
            raise Exception(f"Unknown type {type(item)}")


df = pd.DataFrame(recurse(structure), columns=['from', 'to'])
df

for token in doc:
    print((token.head.text, token.text, token.dep_))

for token in doc:
    print(token.text, token.dep_, token.head.text, token.head.pos_,
            [child for child in token.children])

root = [token for token in doc if token.head == token][0]
subject = list(root.lefts)[0]
for descendant in subject.subtree:
    assert subject is descendant or subject.is_ancestor(descendant)
    print(descendant.text, descendant.dep_, descendant.n_lefts,
            descendant.n_rights,
            [ancestor.text for ancestor in descendant.ancestors])
    
    
edges = []
for token in doc:
    for child in token.children:
        edges.append(('{0}'.format(child.lower_),'{0}'.format(token.lower_)
                      ))
edges
graph = nx.Graph(edges)
G = nx.DiGraph()
G.add_edges_from(edges)

def search(noun):
    a=[]
    for i in range(0,len(edges)):
        if noun in edges[i]:
            a.append(1)
        else:
            a.append(0)
    return a

onde=np.array(list(map(search,select)))

#red_edges = edges[min(np.where(onde[0]==1)[0]):max(np.where(onde[0]==1)[0])]
#edge_colours = ['black' if not edge in red_edges else 'red'
#                for edge in G.edges()]
#black_edges = [edge for edge in G.edges() if edge not in red_edges]
#edgelist=red_edges,
pos = nx.spectral_layout(G)
nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('jet'), node_size = 500)
nx.draw_networkx_labels(G, pos)
nx.draw_networkx_edges(G, pos,  edge_color='r', arrows=False)
nx.draw_networkx_edges(G, pos, arrows=False)
plt.show()

import pandas as pd



### VISUALIZAÇÃO

def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(node.orth_, [to_nltk_tree(child) for child in node.children])
    else:
        return node.orth_
[to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]


import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertModel.from_pretrained('bert-base-uncased')

def embed(word):
    input_ids = tf.constant(tokenizer.encode(word))[None, :]  # Batch size 1
    outputs = model(input_ids)
    last_hidden_states = outputs[0]
    return np.mean(last_hidden_states,axis=1)

embeddings=np.array(list(map(embed,words)))
embeddings=embeddings.reshape(len(words),768)

#! pip install dgl

import dgl


mapping = dict(zip(words,list(range(0,len(words)))))
df= df.replace({'from': mapping, 'to': mapping})

df['from']=df['from'].astype(int)
df['to']=df['to'].astype(int)

def build_graph():
    # All 78 edges are stored in two numpy arrays. One for source endpoints
    # while the other for destination endpoints.
    src = np.array(df['from'])
    dst = np.array(df['to'])
    # Edges are directional in DGL; Make them bi-directional.
    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])
    # Construct a DGLGraph
    return dgl.DGLGraph((u, v))

import torch
import torch.nn as nn
import torch.nn.functional as F
G = build_graph()

embed = nn.Embedding(len(words), embeddings.shape[1])  # 34 nodes with embedding dim equal to 5
G.ndata['feat'] = embed.weight
print(G.ndata['feat'][2])

de=np.array([False]*9)
for i in mask_init:
    de[i-1]=True

g=G
features=G.ndata['feat']
labels= torch.from_numpy(np.array([1,1,1,0,0,0,0,0,0]))
mask=torch.from_numpy(de)



import networkx as nx
# Since the actual graph is undirected, we convert it for visualization
# purpose.
nx_G = G.to_networkx().to_undirected()
# Kamada-Kawaii layout usually looks pretty for arbitrary graphs
pos = nx.kamada_kawai_layout(nx_G)
nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])

from dgl.nn.pytorch import GATConv

class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim):
        super(GATLayer, self).__init__()
        self.g = g
        # equation (1)
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        # equation (1)
        z = self.fc(h)
        self.g.ndata['z'] = z
        # equation (2)
        self.g.apply_edges(self.edge_attention)
        # equation (3) & (4)
        self.g.update_all(self.message_func, self.reduce_func)
        return self.g.ndata.pop('h')
    
class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))

class GAT(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 1)

    def forward(self, h):
        h = self.layer1(h)
        h = F.elu(h)
        h = self.layer2(h)
        return h

from dgl import DGLGraph
from dgl.data import citation_graph as citegrh
import networkx as nx


class Dataset:
    def __init__(self):
        ...
  
    def __getitem__(self, idx):
        """
        Returns
        --------
        DGLGraph
            The i-th graph.
        labels
            The labels for the i-th datapoint.
        """
  
    def __len__(self):
        """
        Returns
        --------
        int
            The size for the dataset.
        """
        
import random

class Subset(object):
    """Subset of a dataset at specified indices
    Code adapted from PyTorch.

    Parameters
    ----------
    dataset
        dataset[i] should return the ith datapoint
    indices : list
        List of datapoint indices to construct the subset
    """
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, item):
        """Get the datapoint indexed by item

        Returns
        -------
        tuple
            datapoint
        """
        return self.dataset[self.indices[item]]

    def __len__(self):
        """Get subset size

        Returns
        -------
        int
            Number of datapoints in the subset
        """
        return len(self.indices)

def k_fold_split(dataset, k, shuffle=True):
    """
    Parameters
    -----------
    dataset
        An instance for the Dataset class defined above.
    k: int
        The number of folds.
    shuffle: bool
        Whether to shuffle the dataset before performing a k-fold split.

    Returns
    --------
    list of length k
        Each element is a tuple (train_set, val_set) corresponding to a fold.
    """
    assert k >= 2, 'Expect the number of folds to be no smaller than 2, got {:d}'.format(k)
    all_folds = []
    indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indices)
    frac_per_part = 1. / k
    data_size = len(dataset)
    for i in range(k):
        val_start = data_size * i * frac_per_part
        val_end = data_size * (i + 1) * frac_per_part
        val_indices = indices[val_start: val_end]
        val_subset = Subset(dataset,  val_indices)
        train_indices = indices[:val_start] + indices[val_end:]
        train_subset = Subset(dataset,  train_indices)
        all_folds.append((train_subset, val_subset))
    return all_folds


# create the model, 2 heads, each head has hidden size 8
net = GAT(g,
          in_dim=features.size()[1],
          hidden_dim=8,
          out_dim=7,
          num_heads=2)

# create optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
import time

dur = []
for epoch in range(30):
    if epoch >= 3:
        t0 = time.time()

    logits = net(features)
    logp = F.log_softmax(logits, 1)
    loss = F.nll_loss(logp[mask], labels[mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch >= 3:
        dur.append(time.time() - t0)

    print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(
        epoch, loss.item(), np.mean(dur)))
    
    

import matplotlib.animation as animation
import matplotlib.pyplot as plt



#for i in range(0,len(pos)):
#    pos[words[i]]=pos.pop(i)

        
def draw(i):
    cls1color = '#00FFFF'
    cls2color = '#FF00FF'
    pos = {}
    colors = []
    for v in range(len(words)):
        pos[v] = all_logits[i][v].numpy()
        cls = pos[v].argmax()
        colors.append(cls1color if cls else cls2color)
    ax.cla()
    ax.axis('off')
    ax.set_title('Epoch: %d' % i)
    nx.draw_networkx(nx_G.to_undirected(), pos,  node_color=colors,
            with_labels=True, node_size=300, ax=ax)
        

for i in range(0,50):
    fig = plt.figure(dpi=150)
    fig.clf()
    ax = fig.subplots()
    draw(i)  # draw the prediction of the first epoch
    plt.savefig('/home/theone/Documents/graph2/foo{}.png'.format(time.time()))
    plt.show()

import glob
from PIL import Image

# filepaths
fp_in = "/home/theone/Documents/graph2/foo*.png"
fp_out = "/home/theone/Documents/graph2_GAT_movie.gif"

# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=200, loop=0)

#ani = animation.FuncAnimation(fig, draw, frames=len(all_logits), interval=200)

#plt.show()


