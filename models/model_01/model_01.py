import torch
import torch.nn as nn

class LNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, num_classes)
        )
    
    def forward(self, x):
        return self.model(x)
    
model = LNN(10, 2)
input = torch.ones((64, 10))
output = model(input)
print(output.shape)    
        