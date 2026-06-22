# ResNet-34 From Scratch

PyTorch implementation of a ResNet-style model for FashionMNIST.

## Files

* `model.py` defines the residual block and ResNet model.
* `data_setup.py` creates FashionMNIST dataloaders.
* `engine.py` contains training and evaluation loops.

## Introduction to ResNet (Residual Network) Architecture

Before diving into the details of the **ResNet** architecture, it is important to understand the challenges researchers faced when training very deep neural networks.

Prior to the development of ResNet, researchers believed that increasing the depth of a neural network would naturally improve its performance, since deeper networks can learn more complex representations. However, experiments showed that simply adding more layers did not always lead to better results.

Researchers observed a phenomenon known as the **degradation problem**. The degradation problem occurs when deeper networks achieve **higher training error** than their shallower counterparts. In theory, a deeper network should perform at least as well as a shallower network because it can learn the same function plus additional transformations. However, optimization becomes increasingly difficult as network depth grows.

Another challenge encountered in deep neural networks is the **vanishing and exploding gradient problem**.

* **Vanishing gradients** occur when gradients become extremely small during backpropagation. As a result, the weights in earlier layers receive very small updates, making learning slow or even preventing the network from learning altogether.
* **Exploding gradients** occur when gradients become excessively large during backpropagation, causing unstable weight updates and making the training process diverge.

Although techniques such as Batch Normalization helped reduce these issues, training very deep networks remained difficult due to the degradation problem.

---

## Residual Networks

Residual Networks (ResNets) were introduced by Kaiming He and colleagues to address the degradation problem and enable the successful training of very deep neural networks.

The key idea behind ResNet is **residual learning**.

Instead of directly learning a complex mapping:

[
H(x)
]

the network learns a residual function:

[
F(x)=H(x)-x
]

which can be rearranged as:

[
H(x)=F(x)+x
]

where:

* (x) is the input to the residual block.
* (H(x)) is the desired underlying mapping.
* (F(x)) is the residual function learned by the network.

This formulation allows the network to focus on learning only the changes that need to be applied to the input rather than learning the entire transformation from scratch.

An important intuition is that if the optimal mapping is close to the identity function, the residual block only needs to learn:

[
F(x)=0
]

which is much easier to optimize than learning the complete transformation directly.

---

## Residual Blocks and Skip Connections

A ResNet is built from multiple **Residual Blocks**.

Each residual block contains:

1. A sequence of convolutional operations.
2. A **skip connection** (also called a shortcut connection) that bypasses these operations.

The input is passed through the convolutional layers to compute the residual function (F(x)), while the skip connection carries the original input (x) directly to the output.

The final output of the block is:

[
y = F(x) + x
]

followed by a ReLU activation function.

If the input and output dimensions differ, a **1×1 projection convolution** is used in the shortcut path to match the dimensions before addition.

A simplified implementation of a residual block is shown below:

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels)
        )

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        out += self.shortcut(x)
        return F.relu(out)
```

---

## Model Architecture

In this project, I re-implemented **ResNet-34**, which consists of 34 learnable layers.

The network is organized into four stages of residual blocks:

### Stage 1

* 3 residual blocks
* Each block contains two 3×3 convolutional layers
* 64 output channels
* Identity shortcut connections

### Stage 2

* 4 residual blocks
* Each block contains two 3×3 convolutional layers
* 128 output channels
* The first block performs downsampling using stride 2
* Projection shortcuts are used when dimensions change

### Stage 3

* 6 residual blocks
* Each block contains two 3×3 convolutional layers
* 256 output channels
* The first block performs downsampling

### Stage 4

* 3 residual blocks
* Each block contains two 3×3 convolutional layers
* 512 output channels
* The first block performs downsampling

### Output Stage

After the final residual stage:

1. Global Average Pooling reduces each feature map to a single value.
2. The resulting feature vector is passed to a fully connected layer.
3. The output logits are used for classification into the FashionMNIST classes.

For training, `CrossEntropyLoss` is used, which applies the necessary softmax operation internally.

# Acknowledgements:
- GeeksforGeeks : https://www.geeksforgeeks.org/deep-learning/residual-networks-resnet-deep-learning/
- Dive into deep learning : https://d2l.ai/chapter_convolutional-modern/resnet.html
- Deep Residual Learning for Image Recognition : https://arxiv.org/abs/1512.03385
