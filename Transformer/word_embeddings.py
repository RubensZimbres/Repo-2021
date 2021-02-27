Using Pytorch:

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
input_ids = torch.tensor(tokenizer.encode("The old Mc Donald has a farm")).unsqueeze(0)
outputs = model(input_ids)
word_embeddings = outputs[0]


Using Tensorflow:

import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertModel.from_pretrained('bert-base-uncased')
input_ids = tf.constant(tokenizer.encode("The old Mc Donald has a farm"))[None, :]
outputs = model(input_ids)
word_embeddings = outputs[0]
