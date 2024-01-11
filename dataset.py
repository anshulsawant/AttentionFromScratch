import datasets as ds
import tokenizers
import numpy as np
import math
from tokenizers import Tokenizer, models, trainers
from jax import numpy as jnp


def load_training_data(n=None):
    '''Could take around 5 minutes to execute.'''
    dataset = ds.load_dataset('wmt14', 'de-en', split='train')
    return dataset if n is None else dataset.select(range(n))


def train_tokenizer(dataset, vocab_size):
    '''Takes a few minutes to execute.'''
    def sentence(dataset):
        ds_iter = dataset.iter(1000)
        for ts in ds_iter:
            for t in ts['translation']:
                yield f"[ST] {t['de']} [EN] [ST] {t['en']} [EN]"
    tokenizer = Tokenizer(models.BPE())
    trainer = trainers.BpeTrainer(
        special_tokens=['[UNK]', '[PAD]', '[ST]', '[EN]'],
        vocab_size=vocab_size,
        min_frequency=3,
        show_progress=True)
    tokenizer.normalizer = tokenizers.normalizers.Sequence(
        [tokenizers.normalizers.NFKC()])
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.BertPreTokenizer()
    tokenizer.train_from_iterator(sentence(dataset), trainer)
    return tokenizer


def augment_dataset_for_training(original, tokenizer, max_length=10, bucket=10):
    indices = np.arange(len(original))
    dataset = original.add_column('index', indices)
    dataset.cleanup_cache_files()
    tokenizer.no_padding()
    tokenizer.enable_truncation(max_length=100)

    def tokenize(translations):
        ts = translations['translation']

        def encode(lang):
            sentences = [f"[ST] {t[lang]} [EN]" for t in ts]
            encoding = tokenizer.encode_batch(sentences)
            tokens = [e.ids for e in encoding]
            return tokens

        en_tokens = encode('en')
        de_tokens = encode('de')
        buckets = [math.ceil(max(len(ets), len(dts))/bucket)*bucket
                   for ets, dts in zip(en_tokens, de_tokens)]

        return {'en_tokens': en_tokens, 'de_tokens': de_tokens,
                'bucket': buckets}

    dataset = dataset.map(tokenize, num_proc=8, batched=True)
    dataset = dataset.remove_columns('translation')
    dataset = dataset.with_format("jax")
    return dataset.sort('bucket')


def training_data_generator(dataset, batch_tokens=25000):
    i = 0
    while i < len(dataset):
        bucket_size = dataset.select([i])[0]['bucket']
        N = batch_tokens//bucket_size
        if i + N > len(dataset):
            N = len(dataset) - i
        bucket_size = dataset.select([i+N-1])[0]['bucket']

        def pad(tokens):
            padding = bucket_size - len(tokens)
            return jnp.pad(tokens, (0, padding))
        batch = dataset.select(range(i, i + N))
        en_tokens = jnp.stack([pad(t) for t in batch['en_tokens']])
        de_tokens = jnp.stack([pad(t) for t in batch['de_tokens']])
        en_mask = en_tokens != 0 
        de_mask = de_tokens != 0
        yield en_tokens, en_mask, de_tokens, de_mask
        i += N


def load_validation_data():
    return ds.load_dataset('wmt14', 'de-en', split='valid')


def load_test_data():
    return ds.load_dataset('wmt14', 'de-en', split='test')


def get_ids(sentence: str, tokenizer):
    return tokenizer.encode(sentence).ids
