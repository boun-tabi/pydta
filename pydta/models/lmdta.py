import re
from functools import lru_cache
from typing import List

import numpy as np
import transformers
from tensorflow.keras.layers import Input, Concatenate, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from transformers import AutoTokenizer, AutoModel

from .dta_model import TFModel


class LMDTA(TFModel):
    def __init__(self, n_epochs=200, learning_rate=0.001, batch_size=256):
        transformers.logging.set_verbosity(transformers.logging.CRITICAL)
        self.chemical_tokenizer = AutoTokenizer.from_pretrained('seyonec/PubChem10M_SMILES_BPE_450k')
        self.chemberta = AutoModel.from_pretrained('seyonec/PubChem10M_SMILES_BPE_450k')

        self.protein_tokenizer = AutoTokenizer.from_pretrained('Rostlab/prot_bert', do_lower_case=False)
        self.protbert = AutoModel.from_pretrained('Rostlab/prot_bert')
        TFModel.__init__(self, n_epochs, learning_rate, batch_size)

    def build(self):
        # Inputs
        chemicals = Input(shape=(768,), dtype='float32')
        proteins = Input(shape=(1024,), dtype='float32')

        # Protein representation
        interaction_representation = Concatenate(axis=-1)([chemicals, proteins])

        # Fully connected layers
        FC1 = Dense(1024, activation='relu')(interaction_representation)
        FC1 = Dropout(0.1)(FC1)
        FC2 = Dense(512, activation='relu')(FC1)
        predictions = Dense(1, kernel_initializer='normal')(FC2)

        opt = Adam(self.learning_rate)
        lmdta = Model(
            inputs=[chemicals, proteins], outputs=[predictions])
        lmdta.compile(optimizer=opt,
                      loss='mean_squared_error',
                      metrics=['mean_squared_error'])
        return lmdta

    @lru_cache(maxsize=2048)
    def get_chemberta_embedding(self, smiles: str):
        tokens = self.chemical_tokenizer(smiles, return_tensors='pt')
        output = self.chemberta(**tokens)
        return output.last_hidden_state.detach().numpy().mean(axis=1)

    def vectorize_chemicals(self, chemicals: List[str]):
        return np.vstack([self.get_chemberta_embedding(chemical) for chemical in chemicals])

    @lru_cache(maxsize=1024)
    def get_protbert_embedding(self, aa_sequence: str):
        pp_sequence = ' '.join(aa_sequence)
        cleaned_sequence = re.sub(r'[UZOB]', 'X', pp_sequence)
        tokens = self.protein_tokenizer(cleaned_sequence, return_tensors='pt')
        output = self.protbert(**tokens)
        return output.last_hidden_state.detach().numpy().mean(axis=1)

    def vectorize_proteins(self, aa_sequences: List[str]):
        return np.vstack([self.get_protbert_embedding(aa_sequence) for aa_sequence in aa_sequences])
