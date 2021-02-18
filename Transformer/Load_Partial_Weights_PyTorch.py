#### Transformer Transfer Learning PyTorch

pretrained_dict = ...
model_dict = model.state_dict()

pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

model_dict.update(pretrained_dict) 

model.load_state_dict(pretrained_dict)
