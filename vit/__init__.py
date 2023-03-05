import torch
import torch.nn as nn
from layers import Tokenization_layer, Transformer

# helper method
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class ViT(nn.Module):
    "Implements Vision Transfromer"
    def __init__(self, *, 
                 image_size,
                 patch_size, 
                 num_classes, 
                 dim, 
                 depth, 
                 heads, 
                 mlp_dim, 
                 pool='cls', 
                 channels=3, 
                 dim_head=64, 
                 dropout=0., 
                 emb_dropout=0.,
                ):
        super().__init__()
        """
        Args:
          image_size (int): the height/weight of the input image.
          patch_size (int): image patch size. In the ViT paper, this value is 16.
          num_classes (num_class): Number of image classes for MLP prediction head.
          dim (int): patch and position embedding dimension.
          depth (int): number of stacked transformer blocks.
          heads (int): number of attention heads.
          mlp_dim (int): inner dimension for MLP in transformer blocks.
          pool (str): choice between "cls" and "mean".
                      For cls, you will need to use the cls token for perdiction
                      For mean, you will need to take the mean of last transformer output 
          channels (int): Input image channels. Set to 3 for RGB image.
          dropout (float): dropout rate for transformer blocks.
          emb_dropout (float): dropout rate for patch embedding.
        
        """
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = 0
        patch_dim = 0
        
        ################# Your Implementations #################################
        # TODO: Compute the num_patches and patch_dim
        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        ################# End of your Implementations ##########################
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        self.pool = pool

        self.to_path_embedding = None

        self.pos_embedding = None
        self.cls_token = None
        self.dropout = None
        self.transformers = nn.ModuleList([])
        self.mlp_head = None
        ################# Your Implementations #################################
        # TODO: 
        # 1) Define self.to_path_embedding using the Tokenization_layer class
        # 2) Define learnable 1-D pos_embedding using torch.randn, the number of 
        #    embedding should be num_patches+1
        # 3) Define learnable 1-D cls_token with dimension = dim. You can use nn.Parameter 
        #    to define the learnable 
        # 4) Define dropout with emb_dropout
        # 5) Define depth num of Transformers
        # 6) Using nn.Sqeuential to create the MLP head including two layers:
        #    The first layer in the MLP head is a LayerNorm layer.
        #    The second layer in the MLP head is a linear layer change dimension to num_classes
        self.to_path_embedding = Tokenization_layer(dim, patch_dim, patch_height, patch_width)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        for i in range(depth):
            self.transformers.add_module(f'transformer{i + 1}', Transformer(dim, heads, dim_head, mlp_dim, dropout))
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        ################# End of your Implementations ##########################

        
    def forward(self, img):
        '''
        Args:
            x (torch.Tensor): input tensor in the shape of (batch_size,N,dim).
        Returns:
            out (torch.Tensor): output tensor in the shape of (batch_size,num_class).
        
        The input tensor 'x' should pass through the following layers:
        1) self.to_patch_embedding: (batch_size,C,H,W) -> (batch_size,N,dim)
        2) Using torch.Tensor.repeat to repeat the cls alone batch dimension.
           Then, concatenate with cls token (batch_size,N,dim) -> (batch_size,N+1,dim)
        3) Take sum of patch embedding and position embedding, then apply dropout. 
        4) Passing through all the transformer blocks (batch_size,N+1,dim) -> (batch_size,N+1,dim)
        5) Use cls token or use pool method to get latent code of batched images 
            (batch_size,N+1,dim) -> (batch_size,dim)
        6) Apply layerNorm to the output of last step
        7) Passing though the final mlp layers: (batch_size,dim) -> (batch_size,num_class)
        
        '''
        out = None
        ################# Your Implementations #################################
        out = self.to_path_embedding(img)
        b, n, _ = out.shape
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        out = torch.cat((cls_tokens, out), dim=1)
        out += self.pos_embedding[:, :(n + 1)]
        out = self.dropout(out)

        # Passing through all the transformer blocks
        for i,transformer in enumerate(self.transformers):
            out = transformer(out)

        # Latent code of batched images
        if self.pool == 'cls':
            out = out[:, 0]
        elif self.pool == 'mean':
            out = out.mean(dim=1)

        # Applying layerNorm + mlp head
        out = self.mlp_head(out)

        ################# End of your Implementations ##########################
        return out