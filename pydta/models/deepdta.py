import numpy as np
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Dense, Dropout, Concatenate
from tensorflow.keras.layers import Input, Embedding
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from .dta_model import TFModel
from ..utils import tokenize_with_hf, encode_smiles


class DeepDTA(TFModel):
    def __init__(self, max_smi_len=100, max_prot_len=1000,
                 embedding_dim=128, learning_rate=0.001,
                 batch_size=256, n_epochs=200,
                 num_filters=32, smi_filter_len=4, prot_filter_len=6):
        print('DeepDTA: Building model')
        self.max_smi_len = max_smi_len
        self.max_prot_len = max_prot_len
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.smi_filter_len = smi_filter_len
        self.prot_filter_len = prot_filter_len

        self.chem_vocab_size = 94
        self.prot_vocab_size = 26
        TFModel.__init__(self, n_epochs, learning_rate, batch_size)
        print('DeepDTA: Model compiled')

    def build(self):
        # Inputs
        chemicals = Input(shape=(self.max_smi_len,), dtype='int32')

        # chemical representation
        chemical_representation = Embedding(input_dim=self.chem_vocab_size + 1,
                                            output_dim=self.embedding_dim,
                                            input_length=self.max_smi_len,
                                            mask_zero=True)(chemicals)
        chemical_representation = Conv1D(filters=self.num_filters,
                                         kernel_size=self.smi_filter_len,
                                         activation='relu',
                                         padding='valid',
                                         strides=1)(chemical_representation)
        chemical_representation = Conv1D(filters=self.num_filters * 2,
                                         kernel_size=self.smi_filter_len,
                                         activation='relu',
                                         padding='valid',
                                         strides=1)(chemical_representation)
        chemical_representation = Conv1D(filters=self.num_filters * 3,
                                         kernel_size=self.smi_filter_len,
                                         activation='relu',
                                         padding='valid',
                                         strides=1)(chemical_representation)
        chemical_representation = GlobalMaxPooling1D()(chemical_representation)

        # Protein representation
        proteins = Input(shape=(self.max_prot_len,), dtype='int32')
        protein_representation = Embedding(input_dim=self.prot_vocab_size + 1,
                                           output_dim=self.embedding_dim,
                                           input_length=self.max_prot_len,
                                           mask_zero=True)(proteins)
        protein_representation = Conv1D(filters=self.num_filters,
                                        kernel_size=self.prot_filter_len,
                                        activation='relu',
                                        padding='valid',
                                        strides=1)(protein_representation)
        protein_representation = Conv1D(filters=self.num_filters * 2,
                                        kernel_size=self.prot_filter_len,
                                        activation='relu',
                                        padding='valid',
                                        strides=1)(protein_representation)
        protein_representation = Conv1D(filters=self.num_filters * 3,
                                        kernel_size=self.prot_filter_len,
                                        activation='relu',
                                        padding='valid',
                                        strides=1)(protein_representation)
        protein_representation = GlobalMaxPooling1D()(protein_representation)

        interaction_representation = Concatenate(axis=-1)([chemical_representation, protein_representation])
        # , axis=-1)

        # Fully connected layers
        FC1 = Dense(1024, activation='relu')(interaction_representation)
        FC1 = Dropout(0.1)(FC1)
        FC2 = Dense(1024, activation='relu')(FC1)
        FC2 = Dropout(0.1)(FC2)
        FC3 = Dense(512, activation='relu')(FC2)
        predictions = Dense(1, kernel_initializer='normal')(FC3)

        opt = Adam(self.learning_rate)
        deepdta = Model(
            inputs=[chemicals, proteins], outputs=[predictions])
        deepdta.compile(optimizer=opt,
                        loss='mean_squared_error',
                        metrics=['mean_squared_error'])
        return deepdta

    def vectorize_chemicals(self, chemicals):
        encoded_smiles = [encode_smiles(smiles) for smiles in chemicals]
        return np.array(tokenize_with_hf('chemical/chembl27_enc_94',
                                         encoded_smiles,
                                         padding_len=self.max_smi_len,
                                         out_type='int'))

    def vectorize_proteins(self, aa_sequences):
        return np.array(tokenize_with_hf('protein/uniprot_26',
                                         aa_sequences,
                                         padding_len=self.max_prot_len,
                                         out_type='int'))
