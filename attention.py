import jax
from jax import numpy as jnp
from jax import random
from jax import nn

def attention_layer(key_dim, value_dim, model_dim, name, batch_size, key = 0):
    key = random.PRNGKey(key)
    ## W_Q is d_model x d_key
    W_Q = random.normal(key, (model_dim, key_dim))
    ## W_K is d_model x d_key
    W_K = random.normal(key, (model_dim, key_dim))
    ## W_V is d_model x d_value
    W_V = random.normal(key, (model_dim, value_dim))
    ## q, k and v are b x n x d_model
    def attention(q, k, v):
        return nn.softmax((q@W_Q)@((k@W_K).transpose((0,2,1))), axis=2)@(v@W_V)

    params = {name: dict(W_Q = W_Q, W_K = W_K, W_V = W_V)}
    return attention, params

def attention_layer_smoke_test():
    key_dim = 2
    value_dim = 4
    model_dim = 3
    name = 'attention'
    key = random.PRNGKey(0)
    x = attention_layer(key_dim, value_dim, model_dim, name)
    q = random.normal(key, (5, model_dim))
    k = random.normal(key, (5, model_dim))
    v = random.normal(key, (5, model_dim))

    att = x[0](q,k,v)

    print(att)
    print(x[1])

def attention_take_apart():
    ## q, k, v are batch size x max input seq length x model dimension
    batch_size = 2
    model_dim = 3
    value_dim = 4
    key_dim = 5
    max_seq_length = 3
    key = random.PRNGKey(0)
    ## These are linear projections from input embeddings to q,k,v.
    W_Q = random.normal(key, (model_dim, key_dim))
    ## W_K is d_model x d_key
    W_K = random.normal(key, (model_dim, key_dim))
    ## W_V is d_model x d_value
    W_V = random.normal(key, (model_dim, value_dim))
                          
    ## q, k and v are b x n x d_model
    q = random.normal(key, (batch_size, max_seq_length, model_dim))
    k = random.normal(key, (batch_size, max_seq_length, model_dim))
    v = random.normal(key, (batch_size, max_seq_length, model_dim))

    ## A cool feature of matrix multiplication in numpy (jax numpy) is that
    ## for 3-d matrices, the matrices along the 0th axis are pair-wise multiplied.
    ## Therefore, if the first dimension is batch size, this just multiplies inputs
    ## for corresponding batch together.
    ## Input sizes: batch_size x max_seq_length x model_dim and model_dim x key_dim
    ## W_Q will be broadcast to size batch_size x model_dim x key_dim
    ## Output size: batch_size x max_seq_length x key_dim
    q_projection = q@W_Q
    ## Similarly, output of size batch_size x max_seq_length x key_dim
    k_projection = k@W_K
    ## Output of size batch_size x max_seq_length x value_dim
    v_projection = v@W_V

    ## Compute attention
    ## When taking transpose, keep batch size dimension as is, only transpose the
    ## other dimensions.
    ## Output of size batch_size x max_seq_length x max_seq_length
    ## Each row tells how important a value is for each query
    A_non_normalized = q_projection@(k_projection.transpose((0,2,1)))
    A = nn.softmax(A_non_normalized, axis=2)

    ## Output is a linear combination of values weighted by importance, as given
    ## by attention matrix A.
    ## E.g., for q_i (i.e., query corresponding to i_th input token), output will be
    ## sum over j (A_ij*v_j)
        
    print(A)

                          
    output = A@v_projection

    print(output)                      
    return (A, output)
                
    
