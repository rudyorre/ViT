import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from einops import rearrange, repeat

class Tokenization_layer(nn.Module):
    def __init__(self, dim, patch_dim, patch_height, patch_width):
        super().__init__()
        """
            Args:
            dim (int): input and output dimension.
            patch_dim(int): flattened vector dimension for image patch
            patch_height (int): height of one image patch
            patch_weight (int): weight of one image patch
            
            You can use Pytorch's built-in function and the above Rearrange method.
            Input and output shapes of each layer:
            1) Rerrange the image: (batch_size, channels, H,W) -> (batch_size,N,patch_dim)
            2) Norm Layer1 (LayerNorm): (batch_size,N,patch_dim) -> (batch_size,N,patch_dim)
            3) Linear Projection layer: (batch_size,N,patch_dim) -> (batch_size,N,dim)
            4) Norm Layer2 (LayerNorm): (batch_size,N,dim)-> (batch_size,N,dim)
        """

        self.to_patch = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width)
        self.norm1 = None 
        self.fc1 = None
        self.norm2 = None

        ################# Your Implementations #################################
        # Hints: You can use the Rearrange method above to achieve faster patch operation
        # Append this layer to nn.Sequential 
        # Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width)
        self.patch_dim = patch_dim
        self.norm1 = nn.LayerNorm(patch_dim)
        self.fc1 = nn.Linear(patch_dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.tokenization_layer = nn.Sequential(
            self.to_patch,
            self.norm1,
            self.fc1,
            self.norm2,
        )
        ################# End of your Implementations ##########################

    def forward(self, x):
        """
        Args:
        x (torch.Tensor): input tensor in the shape of (batch_size,C,H,W)
        Return: 
        out (torch.Tensor): output patch embedding tensor in the shape of (batch_size,N,dim)

        The input tensor 'x' should pass through the following layers:
        1) self.to_patch: Rerrange image 
        2) self.norm1: LayerNorm
        3) self.fc1: Fully-Connected layer
        4) self.norm2: LayerNorm

        """
        ################# Your Implementations #################################
        x = self.tokenization_layer(x)
        ################# End of your Implementations ##########################
        return x
  
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        """
        Args:
          dim (int): input and output dimension.
          heads (int): number of attention heads.
          dim_head (int): input dimension of each attention head.
          dropout (float): dropout rate for attention and final_linear layer.

        Initialize a attention block.
        You can use Pytorch's built-in function.
        Input and output shapes of each layer:
        1) Define the inner dimension as number of heads* dimension of each head
        2) to_qkv: (batch_size, dim) -> (batch_size,3*inner_dimension)
        3) final_linear: (batch_size, inner_dim) -> (batch_size, dim)
        """
        
        self.heads = heads
        self.dim_head = None
        self.to_qkv = None
        self.dropout = None
        
        self.inner_dim = dim_head *  heads    
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        ################# Your Implementations #################################
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=0)
        self.final_linear = nn.Linear(self.inner_dim, dim)
        ################# End of your Implementations ##########################
        
    def forward(self, x):
        '''
        Forward pass of the attention block.
        Args:
            x (torch.Tensor): input tensor in the shape of (batch_size,N,dim).
        Returns:
            out (torch.Tensor): output tensor in the shape of (batch_size,N,dim).
        
        The input tensor 'x' should pass through the following layers:
        1) to_qkv: (batch_size,N,dim) -> (batch_size,N,3*inner_dimension)
        2) Divide the ouput of to qkv to q,k,v and then divide them in to n heads 
            (batch_size,N,inner_dim) -> (batch_size,N,num_head,head_dim)
        3) Use torch.matmul to get the product of q and k
        4) Divide the above tensor by the square root of head dimension
        5) Apply softmax and then dropout on the above tensor
        6) Mutiply the above tensor with v to get attention
        7) Concatenate the attentions from multi-heads 
            (batch_size,N,num_head,head_dim) -> (batch_size,N,inner_dim)
        8) Pass the output from last step to a fully connected layer 
        9) Apply dropout for the last step output    
        '''
        out = None
        
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        ################# Your Implementations #################################
        # Transpose for because of shape: (batch_size, num_heads, query/key, head_dim)
        out = torch.matmul(q, k.transpose(2, 3)) 
        # Divide by sqrt of head dim
        out = out / (self.dim_head ** 0.5)
        # Softmax along the last dimension (key)
        out = F.softmax(out, dim=3)
        out = self.dropout(out)
        out = torch.matmul(out, v)
        # Hint you can use :
        #    out = rearrange(out, 'b h n d -> b n (h d)')
        # to concatenate the output from all attention heads
        # This operation will change the tensor shape from (batch_size,N,num_head,head_dim)
        # to  (batch_size,N,inner_dim)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.final_linear(out)
        out = self.dropout(out)
        ################# End of your Implementations ##########################
        return out
    
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        # keey the resiual connection here
        return self.fn(self.norm(x), **kwargs)+x
    
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, dim, mlp_dim, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        """
         Args:
          dim (int): input and output dimension.
          mlp_dim (int): the output dimension of the fist first layer.
          dropout (float): dropout rate for both linear layers.

        Initialize an MLP.
        You can use Pytorch's built-in nn.Linear function.
        Input and output sizes of each layer:
          1) fc1: dim, mlp_dim
          2) fc2: mlp_dim, dim
        """ 

        self.fc1 = None
        self.fc2 = None
        self.dropout = None
        self.activation = nn.GELU()
        ################# Your Implementations #################################
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, dim)
        self.dropout = nn.Dropout(dropout)
        ################# End of your Implementations ##########################
        
    def forward(self, x):
        '''
        Args:
            x (torch.Tensor): input tensor in the shape of (batch_size,N,dim).
        Returns:
            out (torch.Tensor): output tensor in the shape of (batch_size,N,dim).
        
        The input tensor 'x' should pass through the following layers:
        1) fc1: (batch_size,N,dim) ->  (batch_size,N,mlp_dim)
        2) Apply activation function 
        3) Apply dropout
        3) fc2: (batch_size,N,mlp_dim) -> (batch_size,N,dim)
        4) Apply dropout
        '''
        ################# Your Implementations #################################
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        ################# End of your Implementations ##########################
        return x
    
class Transformer(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout = 0.):
        "Implements Transformer block."
        super().__init__()
        '''
        Args:
          dim (int): input and output dimension.
          heads (int): number of attention heads.
          dim_head (int): input dimension of each attention head.
          mlp_dim (int): 
          dropout (float): dropout rate for attention and FFN layers.
        
        '''
        # Use the PreNorm,Attention and PositionwiseFeedForword class to build your 
        # Transformer block
        self.attn = None
        self.ff = None
        
        ################# Your Implementations #################################
        # a = PreNorm(768, Attention(768, heads = 8, dim_head = 64, dropout = 0.2))
        # self.attn = Attention(dim, heads, dim_head, dropout)
        # self.ff = PositionwiseFeedForward(dim, mlp_dim, dropout)
        self.attn = PreNorm(dim, Attention(dim, heads, dim_head, dropout))
        self.ff = PreNorm(dim, PositionwiseFeedForward(dim, mlp_dim, dropout))
        ################# End of your Implementations ##########################
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): input tensor in the shape of (batch_size,N,dim).
        Returns:
            out (torch.Tensor): output tensor in the shape of (batch_size,N,dim).
        """
        ################# Your Implementations #################################
        residual = x
        x = self.attn(x)
        x += residual
        residual = x
        x = self.ff(x)
        x += residual
        ################# End of your Implementations ##########################
        return x