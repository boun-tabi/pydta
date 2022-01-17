import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor

from ..utils import list_to_numpy


class IDDTA:
    def __init__(self):
        self.prediction_model = DecisionTreeRegressor()
        self.chemical_encoder = OneHotEncoder(handle_unknown='ignore')
        self.protein_encoder = OneHotEncoder(handle_unknown='ignore')

    # def __train_one_hot_encoding(self, train):
    #     chemicals = pd.get_dummies(train['ligand_id'])
    #     # add extra column for cold test
    #     chemicals['cold'] = np.zeros(len(train), int)
    #     self.train_chemicals = chemicals.columns
    #     chemicals = chemicals.values

    #     proteins = pd.get_dummies(train['prot_id'])
    #     # add extra column for cold test
    #     proteins['cold'] = np.zeros(len(train), int)
    #     self.train_proteins = proteins.columns
    #     proteins = proteins.values

    #     return np.hstack([chemicals, proteins])

    # def __test_one_hot_encoding(self, test):
    # column_dict_chem = {k: np.zeros(len(test), int)
    #                     for k in self.train_chemicals}
    # column_dict_prot = {k: np.zeros(len(test), int)
    #                     for k in self.train_proteins}

    # test_ligand_map = {k: (k if k in set(self.train_chemicals) else 'cold')
    #                    for k in test['ligand_id'].unique()}
    # test_prot_map = {k: (k if k in set(self.train_proteins) else 'cold')
    #                  for k in test['prot_id'].unique()}

    # df_c = pd.DataFrame(column_dict_chem)
    # df_c['ligand_id'] = list(test['ligand_id'])
    # df_c['ligand_id'] = df_c['ligand_id'].map(test_ligand_map)
    # for chem in df_c['ligand_id'].unique():
    #     df_c.loc[df_c['ligand_id'] == chem, chem] = 1

    # df_p = pd.DataFrame(column_dict_prot)
    # df_p['prot_id'] = list(test['prot_id'])
    # df_p['prot_id'] = df_p['prot_id'].map(test_prot_map)
    # for prot in df_p['prot_id'].unique():
    #     df_p.loc[df_p['prot_id'] == prot, prot] = 1

    # chemicals = df_c.drop(columns='ligand_id').values
    # proteins = df_p.drop(columns='prot_id').values

    # return np.hstack([chemicals, proteins])

    def vectorize_chemicals(self, chemicals):
        chemicals = np.array(chemicals).reshape(-1, 1)
        return self.chemical_encoder.transform(chemicals).todense()

    def vectorize_proteins(self, proteins):
        proteins = np.array(proteins).reshape(-1, 1)
        return self.protein_encoder.transform(proteins).todense()

    def train(self, train_chemicals, train_proteins, train_labels):
        chemical_vecs = self.chemical_encoder.fit_transform(list_to_numpy(train_chemicals)).todense()
        protein_vecs = self.protein_encoder.fit_transform(list_to_numpy(train_proteins)).todense()

        X_train = np.hstack([chemical_vecs, protein_vecs])
        self.prediction_model.fit(X_train, train_labels)

    def predict(self, chemicals, proteins):
        chemical_vecs = self.vectorize_chemicals(chemicals)
        protein_vecs = self.vectorize_proteins(proteins)
        X_test = np.hstack([chemical_vecs, protein_vecs])
        return self.prediction_model.predict(X_test)
