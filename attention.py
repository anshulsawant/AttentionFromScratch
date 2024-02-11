import jax
from jax import numpy as jnp
from jax import random, nn, lax
import equinox as eqx
from jaxtyping import Array, Bool, Float, PRNGKeyArray, Scalar, Int
import math
from typing import Optional
import dataclasses
import dataset as ds
from typeguard import typechecked
import nltk


@typechecked
class LayerNorm(eqx.Module):
    weight: Float[Array, "..."]
    eps: float
    
    def __init__(self, shape, eps=1e-5):
        # Learnable affine weights
        self.weight = jnp.ones(shape)
        self.eps = eps

    def __call__(self, x: Float[Array, "..."]):
        """Use vmap if normalizing along a particular dimension."""
        mu = jnp.mean(x, keepdims=True)
        sigma = jnp.var(x, keepdims=True)
        return ((x - mu) * jax.lax.rsqrt(sigma + self.eps)) * self.weight


@typechecked
class Embedding(eqx.Module):
    vocab_size: int
    emb_dims: int
    W: Float[Array, "vocab_size emb_dims"]

    def __init__(self, vocab_size, emb_dims, key: PRNGKeyArray):
        self.vocab_size = vocab_size
        self.emb_dims = emb_dims
        self.W = random.normal(key, (vocab_size, emb_dims))

    def __call__(self, x: Scalar) -> Float[Array, "emb_dims"]:
        """Use vmap if indexing by an array."""
        return self.W[x]


@typechecked
class Dropout(eqx.Module):
    p: float

    def __init__(self, p):
        self.p = p

    def __call__(self, x, deterministic: bool, key: PRNGKeyArray):
        if deterministic:
            return x
        keep_p = 1 - self.p
        mask = random.bernoulli(key, p=keep_p, shape=x.shape)
        return lax.select(mask, x / keep_p, jnp.zeros_like(x))


@typechecked
class Linear(eqx.Module):
    in_size: int
    out_size: int
    use_bias: bool
    weight: Float[Array, "out_size in_size"]
    bias: Optional[Float[Array, "out_size"]]

    def __init__(self, in_size: int, out_size: int, key: PRNGKeyArray, use_bias=False):
        wkey, bkey = random.split(key, 2)
        self.in_size = in_size
        self.out_size = out_size
        self.use_bias = use_bias
        lim = 1 / math.sqrt(in_size)
        self.weight = random.uniform(wkey, (out_size, in_size), minval=-lim, maxval=lim)
        self.bias = None
        if self.use_bias:
            self.bias = random.uniform(bkey, (out_size,), minval=-lim, maxval=lim)

    def __call__(self, x: Float[Array, "in_size"]) -> Float[Array, "out_size"]:
        x = self.weight @ x
        if self.use_bias:
            x = x + self.bias
        return x


@typechecked
class DropoutAddNorm(eqx.Module):
    dropout: Dropout
    norm: LayerNorm

    def __init__(self, p, shape):
        self.dropout = Dropout(p)
        self.norm = LayerNorm(shape)

    def __call__(self, x, residue, deterministic, key):
        x = self.dropout(x, deterministic, key)
        x = x + residue
        x = jax.vmap(self.norm)(x)
        return x


@typechecked
class MultiHeadedAttention(eqx.Module):
    emb_dims: int
    nh: int
    q_dim: int
    norm: float
    q_proj: Linear
    k_proj: Linear
    v_proj: Linear
    out_proj: Linear
    post: DropoutAddNorm

    def __init__(self, emb_dims, nh, key, dropout_p):
        assert emb_dims % nh == 0
        # Whether to use causal mask or not.
        self.emb_dims = emb_dims
        self.nh = nh
        self.q_dim = self.emb_dims // self.nh
        self.norm = 1.0 / math.sqrt(self.q_dim)
        q_key, k_key, v_key, out_key = random.split(key, 4)
        self.q_proj = Linear(self.emb_dims, self.emb_dims, q_key, use_bias=False)
        self.k_proj = Linear(self.emb_dims, self.emb_dims, k_key, use_bias=False)
        self.v_proj = Linear(self.emb_dims, self.emb_dims, v_key, use_bias=False)
        self.out_proj = Linear(emb_dims, emb_dims, out_key, use_bias=False)
        self.post = DropoutAddNorm(dropout_p, (emb_dims,))

    def __call__(
        self,
        q: Float[Array, "T emb_dims"],
        k: Float[Array, "T_prime emb_dims"],
        v: Float[Array, "T_prime emb_dims"],
        # Input mask to ignore padded tokens
        deterministic,
        key: PRNGKeyArray,
        # Which inputs to ignore (because of padding or causality)
        mask: Bool[Array, "T T_prime"],
    ) -> Float[Array, "T_prime emb_dims"]:
        Q = jax.vmap(self.q_proj)(q)
        K = jax.vmap(self.k_proj)(k)
        V = jax.vmap(self.v_proj)(v)
        # (nh X T x q_dim)
        Q = jnp.reshape(Q, (-1, self.nh, self.emb_dims // self.nh)).transpose((1, 0, 2))
        # (nh X T' x q_dim)
        K = jnp.reshape(K, (-1, self.nh, self.emb_dims // self.nh)).transpose((1, 0, 2))
        V = jnp.reshape(V, (-1, self.nh, self.emb_dims // self.nh)).transpose((1, 0, 2))
        A = Q @ (K.transpose(0, 2, 1)) * self.norm  # (nh x T x T')
        masked_values = jnp.ones_like(mask) * float("-inf")

        def masking(a):
            return jax.lax.select(mask, a, masked_values)

        A = jax.vmap(masking)(A)
        # softmax axis = -1, (T x emb_dims)
        y = jnp.concatenate(nn.softmax(A) @ V, axis=1)
        y = jax.vmap(self.out_proj)(y)
        return self.post(y, q, deterministic, key)


@typechecked
class FeedForward(eqx.Module):
    linear_1: Linear
    linear_2: Linear
    post: DropoutAddNorm

    def __init__(self, in_dim, inner_dim, dropout_p, key):
        key1, key2 = random.split(key, 2)
        self.linear_1 = Linear(in_dim, inner_dim, key1)
        self.linear_2 = Linear(inner_dim, in_dim, key2)
        self.post = DropoutAddNorm(dropout_p, (in_dim,))

    def __call__(self, x, deterministic, key):
        y = jax.vmap(self.linear_1)(x)
        y = jax.vmap(self.linear_2)(y)
        y = self.post(y, x, deterministic, key)
        return y


@typechecked
class EncoderBlock(eqx.Module):
    mha: MultiHeadedAttention
    ff: FeedForward

    def __init__(self, emb_dims, nh, ff_dim, dropout_p, key):
        mh_key, ff_key = random.split(key, 2)
        self.mha = MultiHeadedAttention(emb_dims, nh, mh_key, dropout_p)
        self.ff = FeedForward(emb_dims, ff_dim, dropout_p, key)

    def __call__(self, x, deterministic, key, mask):
        key_mha, key_ff = random.split(key, 2)
        x = self.mha(x, x, x, deterministic, key_mha, mask)
        x = self.ff(x, deterministic, key_ff)
        return x


@typechecked
class EncoderStack(eqx.Module):
    n_blocks: int
    attention_blocks: list[EncoderBlock]

    def __init__(self, n_blocks, emb_dims, nh, ff_dim, dropout_p, key):
        self.n_blocks = n_blocks
        keys = random.split(key, n_blocks)
        self.attention_blocks = [
            EncoderBlock(emb_dims, nh, ff_dim, dropout_p, key) for key in keys
        ]

    def __call__(self, x, deterministic, key, mask):
        keys = random.split(key, self.n_blocks)
        x = self.attention_blocks[0](x, deterministic, keys[0], mask)
        for ab, key in zip(self.attention_blocks[1:], keys[1:]):
            x = ab(x, deterministic, key, mask)
        return x


@typechecked
class DecoderBlock(eqx.Module):
    self_attention: MultiHeadedAttention
    cross_attention: MultiHeadedAttention
    ff: FeedForward

    def __init__(self, emb_dims, nh, ff_dim, dropout_p, key):
        sa_key, ca_key, ff_key = random.split(key, 3)
        self.self_attention = MultiHeadedAttention(emb_dims, nh, sa_key, dropout_p)
        self.cross_attention = MultiHeadedAttention(emb_dims, nh, ca_key, dropout_p)
        self.ff = FeedForward(emb_dims, ff_dim, dropout_p, ff_key)

    def __call__(self, encoder_out, y, deterministic, key, encoder_mask, causal_mask):
        keys = random.split(key, 3)
        x = self.self_attention(y, y, y, deterministic, keys[0], causal_mask)
        x = self.cross_attention(
            x, encoder_out, encoder_out, deterministic, keys[1], encoder_mask
        )
        x = self.ff(x, deterministic, keys[2])
        return x


@typechecked
class DecoderStack(eqx.Module):
    n_blocks: int
    attention_blocks: list[DecoderBlock]

    def __init__(self, n_blocks, emb_dims, nh, ff_dim, dropout_p, key):
        self.n_blocks = n_blocks
        keys = random.split(key, n_blocks)
        self.attention_blocks = [
            DecoderBlock(emb_dims, nh, ff_dim, dropout_p, key) for key in keys
        ]

    def __call__(self, encoder_out, y, deterministic, key, encoder_mask, causal_mask):
        keys = random.split(key, self.n_blocks)
        x = self.attention_blocks[0](
            encoder_out, y, deterministic, keys[0], encoder_mask, causal_mask
        )
        for ab, key in zip(self.attention_blocks[1:], keys[1:]):
            x = ab(encoder_out, x, deterministic, key, encoder_mask, causal_mask)
        return x


@typechecked
class Transformer(eqx.Module):
    token_embedding: Embedding
    positional_embedding: Embedding
    encoder: EncoderStack
    decoder: DecoderStack
    linear: Linear
    emb_factor: float

    def __init__(
        self,
        vocab_size: int,
        emb_dims: int,
        max_input_seq_length: int,
        n_blocks: int,
        nh: int,
        ff_dim: int,
        dropout_p: float,
        key: PRNGKeyArray,
    ) -> None:
        te_key, pe_key, encoder_key, decoder_key, linear_key = random.split(key, 5)
        self.token_embedding = Embedding(vocab_size, emb_dims, te_key)
        self.positional_embedding = Embedding(max_input_seq_length, emb_dims, pe_key)
        self.encoder: EncoderStack = EncoderStack(
            n_blocks, emb_dims, nh, ff_dim, dropout_p, encoder_key
        )
        self.decoder: DecoderStack = DecoderStack(
            n_blocks, emb_dims, nh, ff_dim, dropout_p, decoder_key
        )
        self.linear: Linear = Linear(emb_dims, vocab_size, linear_key, use_bias=True)
        self.emb_factor: float = math.sqrt(emb_dims)

    def __call__(
        self,
        in_seq: Int[Array, "T_prime"],
        out_seq: Int[Array, "T"],
        deterministic,
        key: PRNGKeyArray,
        encoder_mask: Bool[Array, "T_prime T_prime"],
        cross_attention_mask: Bool[Array, "T T_prime"],
        causal_mask: Bool[Array, "T T"],
    ):
        encoder_key, decoder_key = random.split(key, 2)
        ## T_prime x emb_dims
        in_emb = jax.vmap(self.token_embedding)(in_seq) * self.emb_factor
        ## T x emb_dims
        out_emb = jax.vmap(self.token_embedding)(out_seq) * self.emb_factor
        ## T_prime x emb_dims
        in_pos_emb = jax.vmap(self.positional_embedding)(jnp.arange(in_emb.shape[0]))
        ## T x emb_dims
        out_pos_emb = jax.vmap(self.positional_embedding)(jnp.arange(out_emb.shape[0]))
        ## T_prime x emb_dims
        in_emb = in_emb + in_pos_emb
        ## T x emb_dims
        out_emb = out_emb + out_pos_emb
        ## T_prime x emb_dims
        x = self.encoder(in_emb, deterministic, encoder_key, encoder_mask)
        ## T x emb_dims
        x = self.decoder(
            x, out_emb, deterministic, decoder_key, cross_attention_mask, causal_mask
        )
        print(x.shape)
        ## T x vocab_size
        x = jax.vmap(lambda z: self.token_embedding.W @ z)(x)
        ## T x vocab_size
        return nn.softmax(x, axis=-1)


def bleu_score(model_output, targets):
    """
    Compute the BLEU score for the Transformer model.

    Args:
        model_output (jax.interpreters.xla.DeviceArray): The output from the Transformer model.
        targets (jax.interpreters.xla.DeviceArray): The true labels.

    Returns:
        float: The BLEU score.
    """
    # Convert model output and targets to lists of strings
    model_output = model_output
    targets = targets

    # Compute BLEU score
    return round(nltk.translate.bleu_score.sentence_bleu(targets, model_output), 3)*100


def loss_fn(model_output, targets, label_smoothing=0.1):
    """
    Cross entropy loss function with label smoothing for the Transformer model.

    Args:
        model_output (jax.interpreters.xla.DeviceArray): The output from the Transformer model.
        targets (jax.interpreters.xla.DeviceArray): The true labels.
        label_smoothing (float, optional): The label smoothing factor. Defaults to 0.1.

    Returns:
        jax.interpreters.xla.DeviceArray: The mean loss.
    """
    # Convert targets to one-hot encoding
    targets_one_hot = jax.nn.one_hot(targets, model_output.shape[-1])

    # Apply label smoothing
    targets_one_hot = (
        1 - label_smoothing
    ) * targets_one_hot + label_smoothing / model_output.shape[-1]

    # Compute cross-entropy loss
    loss = -jnp.sum(targets_one_hot * jnp.log(model_output + 1e-10))

    # Return mean loss
    return loss / targets.size


@dataclasses.dataclass
@typechecked
class Config:
    max_input_seq_length: int = 100
    bucket: int = 10
    batch_tokens: int = 25000
    vocab_size: int = 32000
    emb_dims: int = 512
    n_blocks: int = 6
    nh: int = 8
    ff_dim: int = 2048
    dropout_p: float = 0.1
    seed: int = 0
    n: int = None


def build_training_pipeline(config: Config):
    dataset = ds.load_training_data()
    tokenizer = ds.train_tokenizer(dataset, config.vocab_size)
    dataset = ds.augment_dataset_for_training(
        dataset, tokenizer, config.max_input_seq_length, config.bucket
    )
    transformer = Transformer(
        config.vocab_size,
        config.emb_dims,
        config.max_input_seq_length,
        config.n_blocks,
        config.nh,
        config.ff_dim,
        config.dropout_p,
        random.PRNGKey(config.seed),
    )
    data_gen = ds.training_data_generator(dataset, config.batch_tokens)
    return (transformer, data_gen)
