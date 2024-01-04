import torch
from torch import nn
class Net(nn.Module):
    def __init__(self,bias):
        super().__init__()
        self.linear1 = nn.Linear(1024,512)
        self.linear2 = nn.Linear(512,10)
        self.bias = bias
    def forward(self,x,y):
        x_out = self.linear1(x) + 2.0
        y_out = self.linear1(y)
        
        def test_add(x, y):
            return x + 2 + y
        out = test_add(x_out,y_out)
        
        out = self.linear2(out)
        return nn.functional.relu(out)

model = Net(10.)
x = torch.randn(1,1024)
y = torch.randn(1,1024)
torch.onnx.export(model,(x,y),'test_add.onnx',input_names=['x','y'],output_names=['output'])    