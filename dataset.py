import datasets as ds
import tokenizers
from jax import numpy as jnp

def load_training_data():
    '''Could take around 5 minutes to execute.'''
    dataset = ds.load_dataset('wmt14', 'de-en', split='train')
    return dataset


def tokenize(dataset):
    '''Takes a few minutes to execute.'''
    def sentence(dataset):
        ds_iter = dataset.iter(1)
        t = 'translation'
        for s in ds_iter:
            yield f"{s[t][0]['de']} [EN] [ST] {s[t][0]['en']} [EN]"
    tokenizer = tokenizers.CharBPETokenizer()
    tokenizer.add_special_tokens(['[ST]', '[EN]'])
    tokenizer.enable_padding(length=100)
    tokenizer.enable_truncation(max_length=100)
    tokenizer.train_from_iterator(sentence(dataset), vocab_size=37000)

    return tokenizer

def bucket(breaks, dataset):
    lens = [max(len(s['translation'][0]['en']), len(s['translation'][0]['de']))
            for s in dataset.iter(1)]
    indices = argsort(lens)


def load_validation_data():
    return ds.load_dataset('wmt14', 'de-en', split='valid')


def load_test_data():
    return ds.load_dataset('wmt14', 'de-en', split='test')


def get_ids(sentence: str, tokenizer):
    return tokenizer.encode(sentence).ids
