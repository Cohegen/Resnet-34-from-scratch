# ResNet-34 From Scratch

PyTorch implementation of a ResNet-style model for FashionMNIST.

## Files

- `model.py` defines the residual block and ResNet model.
- `data_setup.py` creates FashionMNIST dataloaders.
- `engine.py` contains training and evaluation loops.

## Introduction to Resnet (Residual Network) architecture
- Before diving into the details about **Resnet** architecture, we need to know what challenges researchers were facing.
- Prior to the development of **Resnet** architecture, researchers had a hypothesis that, by adding more layers to a neural network, the accuracy would increase.
- However this didn't go as expected since they hit brick wall called the **degradation problem**. What do we mean when we say our model is facing a degradation problem?What we mean is that as we make networks deeper, training error starts getting worse even though the deeper network should theoretically be able to perform at least as well as a shallower network. The degradation problem occurs when adding more layers causes training accuracy to become worse. In theory, a deeper network should perform at least as well as a shallower one because it could learn the same function plus identity mappings. However, optimization becomes difficult, making deeper networks harder to train.

- Also deeper networks had another disadvantage in that the more layers we have, we encounter an optimization bottleneck. This optimization bottleneck is either vanishing gradients or exploding gradients. Vanishing gradients means that as we perform a backward pass from the output layer to the input layer gradients become so small, to a point they approach zero, that in turn results to very tiny change in weights, which doesn't help us minimize the loss. Exploding gradients occur when gradients become excessively large during backpropagation, causing unstable weight updates and making training diverge.

- Conclusions were made the shallow-layered models would perform better than deeper models, this is because in normal sense shallow-layered models don't face the degradation problem to the level at which deeper models face

### Residual Networks
- Residual networks were  introduce to curb the **degradation problem**.
- A Residual network consists of **Residual blocks**. The innovation here is that, instead of the model learning the underlying complex, exact transformation $H(x)$. Instead of learning the complete transformation H(x), the network learns the residual function F(x)=H(x)−x, which represents the difference between the desired mapping and the input.
- This difference is represented with the equation: $F(x) = H(x)- x$.
- Where:
      - H(x):is our complex,exact transformation.
      - x : is our input image.
      - F(x):is the residual i.e the difference between the transformation and input data.

- This difference is learned by each block in the network. Thus giving making them being named as **Residual networks**.
- The residual block has skip connection which bypasses the layers within the block.
- The input x is added to the output of each block.
- The code of an individual block is shown below:\

```
class ResidualBlock(nn.Module):
  def __init__(self,in_shape,out_shape,stride=1):
    super().__init__()
    self.conv1 = nn.Sequential(
        nn.Conv2d(in_shape,out_shape,kernel_size=3,padding=1,bias=False),
        nn.BatchNorm2d(out_shape),
        nn.ReLU(inplace=True)
    )
    self.conv2 = nn.Sequential(
        nn.Conv2d(out_shape,out_shape,kernel_size=3,padding=1,bias=False),
        nn.BatchNorm2d(out_shape),
       
    )
    #projection shortcut if there's dimension mismatch
    if in_shape != out_shape:
      self.shortcut = nn.Conv2d(
          in_shape,
          out_shape,
          kernel_size=1,
          bias=False
      )
    else:
        self.shortcut = nn.Identity()

  def forward(self,x:torch.Tensor):
    out = self.conv2(self.conv1(x))
    out += self.shortcut(x)
    return F.relu(out)

```

### Model information
- In this project, I have re-implemented resnet-34 meaning our model has 34 layers.
- Our model is divided into several stage:
       * First stage : consists of 3 residual blocks, each with 2 convolution layers of 64 filters and an identity skip connection.
       * Second stage: consists of 4 residual blocks, each with 2 convolution layers of 128 filters, that uses 1x1 projection or padding.
       * Third stage : consists of 6 residual blocks, each with 2 convolution layers of 256 filters.
       * Fourth stage: consists of 3 residual blocks, each with 2 convultion layers of 512 filters.
       * Output stage: where feature maps are passed through a global average pooling followed by a fully connected layer with
       softmax for classification.
