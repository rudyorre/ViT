<p align="center">
<img src='./funny.png' height="300px"/>
</p>

# ViT: Vision Transformer

This is an implementation of the Vision Transformer (ViT) in Pytorch.

The ViT is a transformer-based model for image classification, which achieved competitive performance on image classification benchmarks like ImageNet. It processes an input image as a sequence of patches and then feeds these patches to a transformer-based network.

The above image is a misclassification, however the model was able to reach a 70% accuracy on the validation set only after 20 epochs with a 300 image batch size from a [cat vs dog dataset](https://www.kaggle.com/competitions/dogs-vs-cats-redux-kernels-edition).

## Requirements

- Pytorch >= 1.9.0
- torchvision >= 0.10.0

## Usage

```python
from vit import ViT

model = ViT(
    image_size=224,
    patch_size=16,
    num_classes=1000,
    dim=1024,
    depth=6,
    heads=16,
    mlp_dim=2048,
    pool='cls',
    channels=3,
    dim_head=64,
    dropout=0.1,
    emb_dropout=0.1
)

```

## Arguments

- `image_size` (int): the height/weight of the input image.
- `patch_size` (int): image patch size. In the ViT paper, this value is 16.
- `num_classes` (int): Number of image classes for MLP prediction head.
- `dim` (int): patch and position embedding dimension.
- `depth` (int): number of stacked transformer blocks.
- `heads` (int): number of attention heads.
- `mlp_dim` (int): inner dimension for MLP in transformer blocks.
- `pool` (str): choice between `cls` and `mean`.
- `channels` (int): Input image channels. Set to 3 for RGB image.
- `dim_head` (int): dimension of each attention head
- `dropout` (float): dropout rate for transformer blocks.
- `emb_dropout` (float): dropout rate for patch embedding.

## Methods

- `forward(img)`: Passes the input tensor 'img' through the ViT model to get the output tensor.

## Acknowledgements

- The code is based on the paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" by Dosovitskiy et al. (2021).
