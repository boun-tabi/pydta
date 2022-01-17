import os
import json
import pickle
from functools import lru_cache
from math import sqrt

import numpy as np
from Bio.Align import substitution_matrices
from Bio.pairwise2 import align
from gensim.models import KeyedVectors
from pydta.models.dta_model import BaseDTAModel
from pydta.utils import get_package_path, strip_path
from xgboost import XGBRegressor

DEFAULT_XGB_PARAMS = {
    "random_state": 1,
    "n_jobs": 2,
    "tree_method": "hist",
    "eval_metric": "rmse",
    "subsample": 1,
    "min_child_weight": 10,
    "n_estimators": 2000,
    "learning_rate": 0.1,
    "objective": "reg:squarederror",
    "max_depth": 10,
    "colsample_bytree": 0.9,
}


def load_kmer2vec():
    package_path = get_package_path()
    local_path = f"{package_path}/data/representation"
    if not os.path.exists(f"{local_path}/smilesvec_chembl_8mer.kv"):
        import requests

        print("Downloading SMILESVec kmer embeddings. This can take several minutes.")
        remote_url = (
            "https://cmpe.boun.edu.tr/~riza.ozcelik/pydta/smilesvec_chembl_8mer.zip"
        )
        r = requests.get(remote_url)
        print("Download complete")

        zippath = f"{local_path}/smilesvec_chembl_8mer.zip"
        with open(zippath, "wb") as f:
            f.write(r.content)
        import zipfile

        with zipfile.ZipFile(zippath, "r") as zip_ref:
            zip_ref.extractall(local_path)
        os.remove(zippath)
    print("Loading SMILESVec kmer embeddings...")
    return KeyedVectors.load_word2vec_format(
        f"{local_path}/smilesvec_chembl_8mer.kv", binary=False
    )


class ChemBoost(BaseDTAModel):
    def __init__(self, prot_to_sb_chemicals=None, **kwargs):
        self.kmer2vec = load_kmer2vec()
        self.smilesvec_dim = self.kmer2vec.vector_size

        self.prot_to_sb_chemicals = prot_to_sb_chemicals
        if self.prot_to_sb_chemicals is None:
            self.prot_to_sb_chemicals = {}
        self.train_proteins = set()
        self.sw_scores = dict()
        self.self_sw_scores = dict()

        xgb_params = DEFAULT_XGB_PARAMS
        for k, v in kwargs.items():
            xgb_params[k] = v
        self.xgb = XGBRegressor(**xgb_params)

    @classmethod
    def from_file(cls, path):
        path = strip_path(path)
        instance = cls()
        with open(f"{path}/xgb.pkl", "rb") as f:
            instance.xgb = pickle.load(f)
        with open(f"{path}/train_proteins.pkl", "rb") as f:
            instance.train_proteins = pickle.load(f)
        instance.kmer2vec = load_kmer2vec()

        with open(f"{path}/prot_to_sb_chemicals.json") as f:
            instance.prot_to_sb_chemicals = json.load(f)
        with open(f"{path}/sw_scores.json") as f:
            instance.sw_scores = json.load(f)
        with open(f"{path}/self_sw_scores.json") as f:
            instance.self_sw_scores = json.load(f)
        instance.smilesvec_dim = instance.kmer2vec.vector_size
        return instance

    def get_self_sw_score(self, seq):
        self_sw_score = self.self_sw_scores.get(seq, None)
        if self_sw_score is not None:
            return self_sw_score
        self_sw_score = align.localdx(seq, seq, substitution_matrices.load("blosum62"))[
            0
        ].score
        self.self_sw_scores[seq] = sqrt(self_sw_score)
        return self_sw_score

    def get_sw_score(self, seq1, seq2):
        sw_score = self.sw_scores.get((seq1, seq2), None)
        if sw_score is not None:
            return sw_score
        sw_score = align.localdx(seq1, seq2, substitution_matrices.load("blosum62"))[
            0
        ].score
        sw_score = (
            sw_score / self.get_self_sw_score(seq1) / self.get_self_sw_score(seq2)
        )

        self.sw_scores[(seq1, seq2)] = sw_score
        self.sw_scores[(seq2, seq1)] = sw_score
        return sw_score

    @lru_cache(maxsize=1024)
    def get_sw_vector(self, protein):
        return np.array(
            [
                self.get_sw_score(protein, train_protein)
                for train_protein in self.train_proteins
            ]
        )

    @lru_cache(maxsize=1024)
    def get_ligand_centric_vector(self, protein):
        sb_chemicals = self.prot_to_sb_chemicals.get(protein, [])
        ligand_vecs = [
            self.get_smilesvec_embedding(chemical) for chemical in sb_chemicals
        ]
        if len(ligand_vecs) == 0:
            return np.zeros(self.smilesvec_dim)
        return np.vstack(ligand_vecs).mean(axis=0)

    def vectorize_proteins(self, proteins):
        if len(self.prot_to_sb_chemicals) == 0:
            return np.vstack([self.get_sw_vector(protein) for protein in proteins])
        return np.vstack(
            [
                np.hstack(
                    [
                        self.get_ligand_centric_vector(protein),
                        self.get_sw_vector(protein),
                    ]
                )
                for protein in proteins
            ]
        )

    def get_8mers(self, smiles):
        q = 8
        placeholders = ["D", "E", "J", "X", "j", "t", "z", "x", "d", "R"]
        two_letter_elements = [
            "He",
            "Li",
            "Be",
            "Ne",
            "Na",
            "Mg",
            "Al",
            "Si",
            "Cl",
            "Ar",
            "Ca",
            "Ti",
            "Cr",
            "Mn",
            "Fe",
            "Ni",
            "Cu",
            "Zn",
            "Ga",
            "Ge",
            "As",
            "Se",
            "Br",
            "Kr",
            "Rb",
            "Sr",
            "Zr",
            "Nb",
            "Tc",
            "Ru",
            "Rh",
            "Pd",
            "Ag",
            "Cd",
            "Sb",
            "Te",
            "Xe",
            "Ba",
            "La",
            "Ce",
            "Pr",
            "Nd",
            "Pm",
            "Sm",
            "Eu",
            "Gd",
            "Tb",
            "Dy",
            "Er",
            "Tm",
            "Yb",
            "Lu",
            "Hf",
            "Ta",
            "Re",
            "Ir",
            "Pt",
            "Au",
            "Hg",
            "Tl",
            "Pb",
            "Bi",
            "At",
            "Rn",
            "Fr",
            "Ra",
            "Ac",
            "Th",
            "Pa",
            "Np",
            "Pu",
            "Am",
            "Bk",
            "Es",
            "Fm",
            "Md",
            "Lr",
            "Rf",
            "Db",
            "Sg",
            "Bh",
            "Hs",
            "Mt",
            "Ds",
            "Rg",
            "Uut",
            "Fl",
            "Uup",
            "Lv",
            "Uus",
            "Uuo",
        ]

        def insert_placeholders(smiles):
            el_to_placeholder = {}
            placeholder_count = 0
            for el in two_letter_elements:
                if el in smiles:
                    placeholder = placeholders[placeholder_count]
                    el_to_placeholder[el] = placeholder
                    placeholder_count = placeholder_count + 1
                    smiles = smiles.replace(el, placeholder)
            return smiles, el_to_placeholder

        smiles, mappings = insert_placeholders(smiles)
        lingos = [smiles[ix : ix + q] for ix in range(len(smiles) - (q - 1))]

        def remove_placeholders(lingo, el_to_placeholder):
            for element, placeholder in el_to_placeholder.items():
                lingo = lingo.replace(placeholder, element)
            return lingo

        return [remove_placeholders(lingo, mappings) for lingo in lingos]

    def get_smilesvec_embedding(self, smiles):
        kmers = self.get_8mers(smiles)
        embedding_dim = self.smilesvec_dim
        vecs = []
        for kmer in kmers:
            try:
                vecs.append(self.kmer2vec[kmer])
            except KeyError:
                vecs.append([0.0] * embedding_dim)
        return np.array(vecs).mean(axis=0)

    def vectorize_chemicals(self, chemicals):
        return np.vstack(
            [self.get_smilesvec_embedding(chemical) for chemical in chemicals]
        )

    def train(
        self,
        train_chemicals,
        train_proteins,
        train_labels,
        val_chemicals=None,
        val_proteins=None,
        val_labels=None,
        sample_weights_by_epoch=None,
    ):
        self.train_proteins = set(train_proteins)
        chemical_vecs = self.vectorize_chemicals(train_chemicals)
        protein_vecs = self.vectorize_proteins(train_proteins)
        train_vec = np.hstack([chemical_vecs, protein_vecs])
        val_tuple = None
        if (
            val_chemicals is not None
            and val_proteins is not None
            and val_labels is not None
        ):
            val_chemical_vecs = self.vectorize_chemicals(val_chemicals)
            val_protein_vecs = self.vectorize_proteins(val_proteins)
            val_vec = np.hstack([val_chemical_vecs, val_protein_vecs])
            val_tuple = (val_vec, val_labels())
        self.xgb.fit(train_vec, train_labels, eval_set=val_tuple)

    def predict(self, chemicals, proteins):
        chemical_vecs = self.vectorize_chemicals(chemicals)
        protein_vecs = self.vectorize_proteins(proteins)
        interaction_vecs = np.hstack([chemical_vecs, protein_vecs])
        return self.xgb.predict(interaction_vecs).tolist()

    def save(self, path):
        path = strip_path(path)
        with open(f"{path}/xgb.pkl", "wb") as f:
            pickle.dump(self.xgb, f)
        with open(f"{path}/train_proteins.pkl", "wb") as f:
            pickle.dump(self.train_proteins, f)
        with open(f"{path}/prot_to_sb_chemicals.json", "w") as f:
            json.dump(self.prot_to_sb_chemicals, f)
        with open(f"{path}/sw_scores.json", "w") as f:
            json.dump(self.sw_scores, f)
        with open(f"{path}/self_sw_scores.json", "w") as f:
            json.dump(self.self_sw_scores, f)
