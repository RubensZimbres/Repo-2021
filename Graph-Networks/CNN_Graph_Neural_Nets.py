#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 17:03:53 2021

@author: theone
"""
#nltk.download('averaged_perceptron_tagger')
#!pip install spacy
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


en_nlp = spacy.load('en_core_web_sm')
phrase="The quick brown fox jumps over the lazy dog"
doc = en_nlp(phrase)
root=[i.root for i in doc.sents]

words=word_tokenize(phrase)

for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
            token.shape_, token.is_alpha, token.is_stop)

select=[]
for token in doc:
    if token.pos_=='PROPN' or token.pos_=='NOUN':
        print(token.text,token.pos_)
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

import networkx as nx
# Since the actual graph is undirected, we convert it for visualization
# purpose.
nx_G = G.to_networkx().to_undirected()
# Kamada-Kawaii layout usually looks pretty for arbitrary graphs
pos = nx.kamada_kawai_layout(nx_G)
nx.draw(nx_G, pos, with_labels=True, node_color=[[.7, .7, .7]])

from dgl.nn.pytorch import GraphConv

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, num_classes)

    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = torch.relu(h)
        h = self.conv2(g, h)
        return h

# The first layer transforms input features of size of 5 to a hidden size of 5.
# The second layer transforms the hidden layer and produces output features of
# size 2, corresponding to the two groups of the karate club.
net = GCN(768, 5, 2)

inputs = embed.weight
labeled_nodes = torch.tensor([0, len(words)-1])  # only the instructor and the president nodes are labeled
labels = torch.tensor([0, 1])  # their labels are different

import itertools

optimizer = torch.optim.Adam(itertools.chain(net.parameters(), embed.parameters()), lr=0.01)
all_logits = []
for epoch in range(50):
    logits = net(G, inputs)
    # we save the logits for visualization later
    all_logits.append(logits.detach())
    logp = F.log_softmax(logits, 1)
    # we only compute loss for labeled nodes
    loss = F.nll_loss(logp[labeled_nodes], labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))


import matplotlib.animation as animation
import matplotlib.pyplot as plt

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
    nx.draw_networkx(nx_G.to_undirected(), pos, node_color=colors,
            with_labels=True, node_size=300, ax=ax)

fig = plt.figure(dpi=150)
fig.clf()
ax = fig.subplots()
draw(0)  # draw the prediction of the first epoch
plt.show()


#ani = animation.FuncAnimation(fig, draw, frames=len(all_logits), interval=200)

#plt.show()




















