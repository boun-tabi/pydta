import numpy as np
from sklearn.tree import DecisionTreeRegressor
from tensorflow.keras.preprocessing.text import Tokenizer

from ..utils import encode_smiles, tokenize_with_hf


class BoWDTA:
    def __init__(self):
        self.chemical_bow_vectorizer = Tokenizer(
            filters=None, lower=False, oov_token='C')
        self.protein_bow_vectorizer = Tokenizer(
            filters=None, lower=False, oov_token='$')
        self.prediction_model = DecisionTreeRegressor()

    def tokenize_chemicals(self, smiles):
        encoded_smiles = [encode_smiles(s) for s in smiles]
        return tokenize_with_hf('chemical/chembl27_enc_bpe_8000',
                                encoded_smiles,
                                out_type='int',
                                padding_len=100)

    def tokenize_proteins(self, proteins):
        return tokenize_with_hf('protein/uniprot_bpe_32000',
                                proteins,
                                out_type='int',
                                padding_len=1000)

    def vectorize_chemicals(self, encoded_smiles_words):
        return self.chemical_bow_vectorizer.texts_to_matrix(encoded_smiles_words, mode='freq')

    def vectorize_proteins(self, proteins):
        return self.protein_bow_vectorizer.texts_to_matrix(proteins, mode='freq')

    def train(self, train_chemicals, train_proteins, train_labels):
        tokenized_chemicals = self.tokenize_chemicals(train_chemicals)
        tokenized_proteins = self.tokenize_proteins(train_proteins)
        self.chemical_bow_vectorizer.fit_on_texts(tokenized_chemicals)
        self.protein_bow_vectorizer.fit_on_texts(tokenized_proteins)

        chemical_vectors = self.vectorize_chemicals(tokenized_chemicals)
        protein_vectors = self.vectorize_proteins(tokenized_proteins)
        X_train = np.hstack([chemical_vectors, protein_vectors])
        self.prediction_model.fit(X_train, train_labels)

    def predict(self, chemicals, proteins):
        tokenized_chemicals = self.tokenize_chemicals(chemicals)
        tokenized_proteins = self.tokenize_proteins(proteins)

        chemical_vectors = self.vectorize_chemicals(tokenized_chemicals)
        protein_vectors = self.vectorize_proteins(tokenized_proteins)

        interaction = np.hstack([chemical_vectors, protein_vectors])
        return self.prediction_model.predict(interaction).tolist()
