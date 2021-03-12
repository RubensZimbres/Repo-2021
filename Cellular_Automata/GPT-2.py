import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Model, GPT2Config
import pandas as pd

df=pd.read_csv('/home/anaconda3/work/goodreads_books.csv',sep=',',header=0)

x_train = df.description.dropna()[0:10]
df['average_rating'][df['average_rating']<2.5]=0
df['average_rating'][df['average_rating']>2.5]=1
y_train = df['average_rating'][0:10]

from transformers import BertModel, BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def embed(x):
    input_ids = torch.tensor(tokenizer.encode(x)).unsqueeze(0)  # Batch size 1
    return torch.sum(model(input_ids)[0],(1))

x=torch.cat([embed(str(x_train.iloc[i])) for i in range(0,10)])

x.shape

y=torch.tensor(y_train)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def generate_example(n):
    bits = np.random.randint(low=0, high=2, size=(2, n))
    xor = np.logical_xor(bits[0], bits[1]).astype(np.long)
    return bits.reshape((2*n)), xor


configuration = GPT2Config()
model = GPT2Model(configuration)
configuration = model.config


gpt2 = GPT2Model.from_pretrained('gpt2')
in_layer = nn.Embedding(2, 768)
out_layer = nn.Linear(768, 2) 

for name, param in gpt2.named_parameters():
    # freeze all parameters except the layernorm and positional embeddings
    if 'ln' in name or 'wpe' in name:
        param.requires_grad = True
    else:
        param.requires_grad = False


params = list(gpt2.parameters()) + list(in_layer.parameters()) + list(out_layer.parameters())
optimizer = torch.optim.Adam(params)
loss_fn = nn.CrossEntropyLoss()

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'


for layer in (gpt2, in_layer, out_layer):
    layer.to(device=device)
    layer.train()

accuracies = [0]
n=5
while sum(accuracies[-50:]) / len(accuracies[-50:]) < .99:
    x, y = generate_example(n)
    x = torch.from_numpy(x).to(device=device, dtype=torch.long)
    y = torch.from_numpy(y).to(device=device, dtype=torch.long)
    
    embeddings = in_layer(x.reshape(1, -1))
    hidden_state = gpt2(inputs_embeds=embeddings).last_hidden_state[:,n:]
    logits = out_layer(hidden_state)[0]
    
    loss = loss_fn(logits, y)
    accuracies.append((logits.argmax(dim=-1) == y).float().mean().item())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if len(accuracies) % 500 == 0:
        accuracy = sum(accuracies[-50:]) / len(accuracies[-50:])
        print(f'Samples: {len(accuracies)}, Accuracy: {accuracy}')

print(f'Final accuracy: {sum(accuracies[-50:]) / len(accuracies[-50:])}')

for layer in (gpt2, in_layer, out_layer):
    layer.eval()


with torch.no_grad():
    x = torch.from_numpy(bits).to(device=device, dtype=torch.long)
    
    embeddings = in_layer(x)
    transformer_outputs = gpt2(
        inputs_embeds=embeddings,
        return_dict=True,
        output_attentions=True,
    )
    logits = out_layer(transformer_outputs.last_hidden_state[n:])
    predictions = logits.argmax(dim=-1).cpu().numpy()

print('  String 1:', bits[:n])
print('  String 2:', bits[n:])
print('Prediction:', predictions)
print('Output XOR:', xor)

attentions = transformer_outputs.attentions[0][0]  # first layer, first in batch
mean_attentions = attentions.mean(dim=0)           # take the mean over heads
mean_attentions = mean_attentions.cpu().numpy()

plt.xlabel('Input Tokens', size=16)
plt.xticks(range(10), bits)
plt.ylabel('Output Tokens', size=16)
plt.yticks(range(10), ['*'] * 5 + list(predictions))
plt.imshow(mean_attentions)
plt.show()

# Sanity check

fresh_gpt2 = GPT2Model.from_pretrained('gpt2')

gpt2.to(device='cpu')
gpt2_state_dict = gpt2.state_dict()
for name, param in fresh_gpt2.named_parameters():
    if 'attn' in name or 'mlp' in name:
        new_param = gpt2_state_dict[name]
        if torch.abs(param.data - new_param.data).sum() > 1e-8:
            print(f'{name} was modified')
        else:
            print(f'{name} is unchanged')

##########################################################################

import tensorflow as tf
from transformers import GPT2Tokenizer, TFGPT2DoubleHeadsModel


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = TFGPT2DoubleHeadsModel.from_pretrained('gpt2')

num_added_tokens = tokenizer.add_special_tokens({'cls_token': '[CLS]'})
embedding_layer = model.resize_token_embeddings(len(tokenizer))

choices = ["Hello, my dog is cute [CLS]", "Hello, my cat is cute [CLS]"]
encoded_choices = [tokenizer.encode(s) for s in choices]


cls_token_location = [tokens.index(tokenizer.cls_token_id) for tokens in encoded_choices]
input_ids = tf.constant(encoded_choices)[None, :]  # Batch size: 1, number of choices: 2

mc_token_ids = tf.constant([cls_token_location])  # Batch size: 1
outputs = model(input_ids, mc_token_ids=mc_token_ids)
lm_prediction_scores, mc_prediction_scores = outputs[:2]



########################################################################

from transformers import GPT2Tokenizer, GPT2Model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)

#####################################

## GENERATE
from transformers import TFGPT2LMHeadModel, GPT2Tokenizer

model = TFGPT2LMHeadModel.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

input_ids = tokenizer.encode('I enjoy walking with my cute dog', return_tensors='tf')

greedy_output = model.generate(input_ids, max_length=50)

beam_output = model.generate(
    input_ids, 
    max_length=50, 
    num_beams=5, 
    no_repeat_ngram_size=2, 
    early_stopping=True,top_p=0.92, 
    top_k=0
)

print(tokenizer.decode(beam_output[0], skip_special_tokens=True))



#######################################
