from transformers import SqueezeBertTokenizer, SqueezeBertModel
import torch

tokenizer = SqueezeBertTokenizer.from_pretrained('squeezebert/squeezebert-mnli-headless')
model = SqueezeBertModel.from_pretrained('squeezebert/squeezebert-mnli-headless')

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state
