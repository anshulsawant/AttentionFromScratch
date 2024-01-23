from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from attention import Transformer
from attention import bleu_score

# Create a random key for testing
key = jax.random.PRNGKey(0)

# Initialize the Transformer model
vocab_size = 10
emb_dims = 32
max_input_seq_length = 10
n_blocks = 6
nh = 8
ff_dim = 64
dropout_p = 0.1
deterministic: bool = False
key = jax.random.PRNGKey(0)
transformer = Transformer(
    vocab_size, emb_dims, max_input_seq_length, n_blocks, nh, ff_dim, dropout_p, key
)

# Generate some input sequences for testing
batch_size = 2
input_seq_length = 5
output_seq_length = 6
in_seq = np.random.randint(0, vocab_size, (batch_size, input_seq_length))
out_seq = np.random.randint(0, vocab_size, (batch_size, output_seq_length))

# Generate masks for testing
encoder_mask = np.ones((batch_size, input_seq_length, input_seq_length), dtype=bool)
cross_attention_mask = np.ones(
    (batch_size, output_seq_length, input_seq_length), dtype=bool
)
causal_mask = (
    np.tril(np.ones((batch_size, output_seq_length, output_seq_length)), k=1) == 1
)


# Test the __call__ method of the Transformer model
def f(in_seq, out_seq, encoder_mask, cross_attention_mask, causal_mask):
    return transformer(
        in_seq,
        out_seq,
        deterministic,
        key,
        encoder_mask,
        cross_attention_mask,
        causal_mask,
    )


output = jax.vmap(f)(in_seq, out_seq, encoder_mask, cross_attention_mask, causal_mask)
print(output.shape)  # Expected output: (batch_size, output_seq_length, vocab_size)
import nltk
import jax.numpy as jnp
import numpy as np

def test_bleu_score():
    # Test inputs
    model_output = [1, 2, 3, 4, 6, 7, 8]
    targets = [[1, 2, 3, 4, 5, 6, 7, 8]]

    # Expected output
    expected_bleu_score = 51.5 

    actual_score = bleu_score(model_output, targets)
    print("Expected score: ", expected_bleu_score)
    print("Actual score: ", actual_score)
    # Test the bleu_score function
    assert actual_score == expected_bleu_score

test_bleu_score()