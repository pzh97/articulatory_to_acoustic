import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  
        self.out = torch.nn.Linear(n_hidden, n_output)  

    def forward(self, x):
        x = F.relu(self.hidden(x))     
        x = self.out(x)
        return x

net = Net(n_feature=22, n_hidden=100, n_output=81)
net = net.float()

