import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
  def __init__(self, in_shape, out_shape, stride=1):
    super().__init__()
    self.conv1 = nn.Sequential(
        nn.Conv2d(in_shape, out_shape, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_shape),
        nn.ReLU(inplace=True)
    )
    self.conv2 = nn.Sequential(
        nn.Conv2d(out_shape, out_shape, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_shape),
    )
    if stride != 1 or in_shape != out_shape:
      self.shortcut = nn.Sequential(
          nn.Conv2d(in_shape, out_shape, kernel_size=1, stride=stride, bias=False),
          nn.BatchNorm2d(out_shape),
      )
    else:
        self.shortcut = nn.Identity()

  def forward(self,x:torch.Tensor):
    out = self.conv2(self.conv1(x))
    out += self.shortcut(x)
    return F.relu(out)

class Resnet(nn.Module):
  def __init__(self,num_classes):
    super().__init__()
    ## input layer
    self.conv1 = nn.Sequential(
        nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),
       
    )

    ##stage 1:  3 residual blocks
    self.stage = nn.Sequential(
        ResidualBlock(64,64),
        ResidualBlock(64,64),
        ResidualBlock(64,64)
    )

    ##stage 2 : 4 residual blocks
    self.stage2 = nn.Sequential(
        ResidualBlock(64,128,stride=2),
        ResidualBlock(128,128),
        ResidualBlock(128,128),
        ResidualBlock(128,128)
    )

    ##stage3 : 6 residual blocks
    self.stage3 = nn.Sequential(
        ResidualBlock(128,256,stride=2),
        ResidualBlock(256,256),
        ResidualBlock(256,256),
        ResidualBlock(256,256),
        ResidualBlock(256,256),
        ResidualBlock(256,256)
    )

    ##stage 4 : 3 residual block
    self.stage4 = nn.Sequential(
        ResidualBlock(256,512,stride=2),
         ResidualBlock(512,512),
         ResidualBlock(512,512),
        
        
    )
    

   

    ## final output
    self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    self.fc = nn.Linear(512,num_classes)


  def forward(self,x:torch.Tensor):
   x = self.stage4(self.stage3(self.stage2(self.stage(self.conv1(x)))))
   ##final out
   x_avg = self.avgpool(x)
   x_avg = torch.flatten(x_avg,1)
   x_final = self.fc(x_avg)

   return x_final
