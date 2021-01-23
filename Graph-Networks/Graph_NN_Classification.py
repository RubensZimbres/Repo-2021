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

embeddings=norm(np.array(list(map(embed,words))))
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

import torch.nn.functional as f
G = build_graph()
g=G

g.in_degrees().view(-1, 1).float()
features=f.normalize(torch.tensor(embeddings), p=2, dim=1)

labels= torch.from_numpy(np.array([1,1,1,0,0,0,0,0,0]))
mask=torch.from_numpy(de)


###################################################################################


from dgl.data import MiniGCDataset
import matplotlib.pyplot as plt
import networkx as nx
graph=g
label=labels
fig, ax = plt.subplots()
nx.draw(graph.to_networkx(), ax=ax)
plt.show()

import dgl
import torch

def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)

from dgl.nn.pytorch import GraphConv

import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv


class Classifier(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(Classifier, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, num_classes)

    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = torch.relu(h)
        h = self.conv2(g, h)
        return h
    
import torch.optim as optim
from torch.utils.data import DataLoader

import pandas as pd
import torch
import random
from torch.utils.data import Dataset, TensorDataset

inps = features
tgts = labels
dataset = torch.utils.data.TensorDataset(inps, tgts)



trainset = TensorDataset(inps,tgts)
data_loader = torch.utils.data.DataLoader(trainset, batch_size=9,shuffle=True)

import torch.optim as optim
from torch.utils.data import DataLoader


model = Classifier(768, 20, 2)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.train()


epoch_losses = []
for epoch in range(10):
    epoch_loss = 0
    for iter, (bg, label) in list(enumerate(data_loader)):
        prediction = model(G,bg)
        loss = loss_func(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
    epoch_loss /= (iter + 1)
    print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
    epoch_losses.append(epoch_loss)

plt.title('cross entropy averaged over minibatches')
plt.plot(epoch_losses)
plt.show()


#!pip install sklearn scikit-learn

torch.save(model.state_dict(), '/home/theone/Documents/Model1/checkpoint.t7')

model = Classifier(768, 20, 2)


model.load_state_dict(torch.load('/home/theone/Documents/Model1/checkpoint.t7'))
model.eval()

import time        
def draw(i):
    cls1color = '#FF3333'
    cls2color = '#99FF33'
    pos = {}
    colors = []
    for v in range(len(words)):
        pos[v] = prediction[v].detach().numpy()
        cls = pos[v].argmax()
        colors.append(cls1color if cls else cls2color)
    ax.cla()
    ax.axis('off')
    ax.set_title('Epoch: %d' % i)
    nx.draw(graph.to_networkx(), ax=ax,node_color='red')
        
draw(0)

for i in range(0,50):
    fig = plt.figure(dpi=150)
    fig.clf()
    ax = fig.subplots()
    draw(i)  # draw the prediction of the first epoch
    plt.savefig('/home/theone/Documents/graph3/foo{}.png'.format(time.time()))
    plt.show()

import glob
from PIL import Image

# filepaths
fp_in = "/home/theone/Documents/graph3/foo*.png"
fp_out = "/home/theone/Documents/graph3_Class_movie.gif"

# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
img.save(fp=fp_out, format='GIF', append_images=imgs,
         save_all=True, duration=200, loop=0)


