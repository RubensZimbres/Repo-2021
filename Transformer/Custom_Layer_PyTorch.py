class Custom(nn.Module):
    def __init__(self, d_model, vocab):
        super(Custom, self).__init__()
        self.d_model = d_model
        self.w_1 = nn.Linear(vocab,d_model)

    def forward(self, x):
        return self.w_1(x.view(-1,x.size(0))) * math.sqrt(self.d_model)
    
########################################################################

class Custom(nn.Module):
    def __init__(self, d_model, vocab):
        super(Custom, self).__init__()
        self.d_model = d_model
 
    def forward(self, x):
        return torch.cat(4*[x]).reshape(3772,8,self.d_model)  * math.sqrt(self.d_model)

