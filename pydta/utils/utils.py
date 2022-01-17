import json
import pathlib
import re

import numpy as np
from tokenizers import Tokenizer


def get_package_path():
    return str(pathlib.Path(__file__).parent.parent.resolve())


package_path = get_package_path()


class HFWordIdentifier:
    def __init__(self, path):
        self.tokenizer = Tokenizer.from_file(path)

    def identify_words(self, sequences, padding_len=None, out_type='int'):
        encodings = self.tokenizer.encode_batch(sequences)
        if padding_len is not None:
            for encoding in encodings:
                encoding.pad(padding_len, direction='right',
                             pad_id=0, pad_token='[PAD]')
                encoding.truncate(padding_len)

        if out_type == 'int':
            return [encoding.ids for encoding in encodings]
        elif out_type == 'str':
            return [encoding.tokens for encoding in encodings]
        else:
            raise ValueError('Invalid out_type for word identification')


def tokenize_with_hf(hf_tokenizer_name, sequences,
                     padding_len, out_type='int'):
    vocabs_path = f'{package_path}/data/vocabs'
    tokenizer_path = f'{vocabs_path}/{hf_tokenizer_name}.json'
    # tokenizer = HFWordIdentifier.from_file(tokenizer_path)
    tokenizer = HFWordIdentifier(tokenizer_path)
    return tokenizer.identify_words(sequences,
                                    padding_len=padding_len,
                                    out_type=out_type)


def smiles_segmenter(smi):
    pattern = '(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])'
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    return tokens


def encode_smiles(smiles):
    segments = smiles_segmenter(smiles)
    vocabs_path = f'{package_path}/data/vocabs'
    with open(f'{vocabs_path}/chemical/chembl27_encoding.json') as f:
        encoding_vocab = json.load(f)

    return ''.join([encoding_vocab.get(segment, encoding_vocab['[OOV]']) for segment in segments])


def strip_path(path):
    if path.endswith('/'):
        return path[:-1]


def create_uniform_weights(n_samples, n_epochs):
    return [np.array([1] * n_samples) for _ in range(n_epochs)]


def list_to_numpy(lst):
    return np.array(lst).reshape(-1, 1)


def load_sample_dta_data(mini=False):
    if mini:
        sample_data_path = f'{package_path}/data/dta_sample_data/dta_sample_data_mini.json'
    else:
        sample_data_path = f'{package_path}/data/dta_sample_data/dta_sample_data.json'
    with open(sample_data_path) as f:
        return json.load(f)
