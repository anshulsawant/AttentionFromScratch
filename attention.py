import jax
from jax import numpy as jnp
from jax import random, nn, lax
import equinox as eqx
import jaxtyping
from jaxtyping import Float
from jaxtyping import Array
from jaxtyping import Scalar
from jaxtyping import PRNGKeyArray
import math

class Config:
    T = 128 # Max input sequence length, timesteps
    batch_size = 100
    D = 512 # Embedding dimension
    nh = 8 # Number of heads
    eps_ls = 0.1 # Label smoothing
    dropout_p = 0.1 # Dropout rate
    feed_forward_dim = 2048
    adam_beta1 = 0.9
    adam_beta2 = 0.98
    adam_eps = 1e-9
    n_encoder_blocks = 6
    n_decoder_blocks = 6
    
class LayerNorm(eqx.Module):
    weight: Float[Array , '...']
    eps: float
    
    def __init__(self, shape, eps=1e-5):
        ## Learnable affine weights
        self.weight = jnp.ones(shape)
        self.eps = eps

    def __call__(self, x: Float[Array, "..."]):
        """Use vmap if normalizing along a particular dimension."""
        mu = jnp.mean(x, keepdims=True)
        sigma = jnp.var(x, keepdims=True)
        return ((x - mu) * jax.lax.rsqrt(sigma + self.eps))*self.weight
        

class Embedding(eqx.Module):
    def __init__(self, vocab_size, emb_dims, key: PRNGKeyArray):
        self.vocab_size = vocab_size
        self.emb_dims = emb_dims
        self.W = random.normal(key, (vocab_size, emb_dims))

    def __call__(self, x: Scalar) -> Float[Array, "{self.emb_dims}"]:
        """Use vmap if indexing by an array."""
        return self.W[x]
        

class Dropout(eqx.Module):
    def __init__(self, p):
        self.p = p
        
    def __call__(x, deterministic: bool, key: PRNGKeyArray):
        if deterministic:
            return x
        keep_p = 1 - self.p
        mask = random.bernoulli(key, p=keep_p, shape=x.shape)
        return lax.select(mask, x/keep_p, jnp.zeros_like(x))
        

class Linear(eqx.Module):
    in_size: int
    out_size: int
    use_bias: bool
    weight: Float[Array, '...']
    
    def __init__(self, in_size: int, out_size: int, key: PRNGKeyArray, use_bias=True):
        wkey, bkey = random.split(key, 2)
        self.in_size = in_size
        self.out_size = out_size
        self.use_bias = use_bias
        lim = 1/math.sqrt(in_size)
        self.weight = random.uniform(wkey, (out_size, in_size), minval=-lim, maxval=lim)
        if self.use_bias:
            self.bias = random.uniform(bkey, (out_size,), minval=-lim, maxval=lim)

    def __call__(self, x: Float[Array, "in_size"]) -> Float[Array, "out_size"]:
        x = self.weight @ x
        if self.use_bias:
            x = x + self.bias
        return x
        
    
class MultiHeadedAttention(eqx.Module):
    masked: bool
    emb_dims: int
    nh: int
    q_dim: int
    norm: float
    input_proj: Linear
    out_proj: Linear
    
    def __init__(self, emb_dims, nh, key, masked=False):
        assert emb_dims % nh == 0
        self.masked = masked
        self.emb_dims = emb_dims
        self.nh = nh
        self.q_dim = self.emb_dims // self.nh
        self.norm = 1.0/math.sqrt(self.q_dim)
        in_key, out_key = random.split(key)
        self.input_proj = Linear(self.emb_dims, 3*self.emb_dims, in_key, use_bias=False)
        self.out_proj = Linear(emb_dims, emb_dims, out_key, use_bias=False)
        
    def __call__(self,
                 x: Float[Array, "T {self.emb_dims}"]) -> Float[Array, "T {self.emb_dims}"]:
        T = x.shape[0]
        Q, K, V = jnp.split(jax.vmap(self.input_proj)(x), 3, -1)
        # (nh X T x q_dim)
        Q = jnp.reshape(Q, (T, self.nh, self.emb_dims//self.nh)).transpose((1, 0, 2))
        K = jnp.reshape(K, (T, self.nh, self.emb_dims//self.nh)).transpose((1, 0, 2))
        V = jnp.reshape(V, (T, self.nh, self.emb_dims//self.nh)).transpose((1, 0, 2))
        A = Q@(K.transpose(0, 2, 1)) * self.norm ## (nh x T x T)
        if self.masked:
            mask = jnp.tri(A.shape[1]) != 0
            masked_values = jnp.ones_like(mask)*float('-inf')
            def masking(a):
                return jax.lax.select(mask, a, masked_values)
            A = jax.vmap(masking)(A)
        y = jnp.concatenate(nn.softmax(A)@V, axis=1) # softmax axis = -1, (T x emb_dims)
        return jax.vmap(self.out_proj)(y)
        
        
