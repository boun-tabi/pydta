import json
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf

from ..evaluation import evaluate_predictions
from ..utils import create_uniform_weights, strip_path

tf.get_logger().setLevel('WARNING')


class BaseDTAModel(ABC):
    @abstractmethod
    def train(self, train_chemicals, train_proteins, train_labels,
              val_chemicals=None, val_proteins=None, val_labels=None):
        pass

    @abstractmethod
    def predict(self, chemicals, proteins):
        pass


class TFModel(BaseDTAModel):
    @abstractmethod
    def __init__(self, n_epochs, learning_rate, batch_size, **kwargs):
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.history = {}
        self.model = self.build()

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def vectorize_chemicals(self, chemicals):
        pass

    @abstractmethod
    def vectorize_proteins(self, proteins):
        pass

    @classmethod
    def from_file(cls, path):
        with open(f'{path}/params.json') as f:
            dct = json.load(f)

        instance = cls(**dct)

        instance.model = tf.keras.models.load_model(f'{path}/model')

        with open(f'{path}/history.json') as f:
            instance.history = json.load(f)
        return instance

    def train(self, train_chemicals, train_proteins, train_labels,
              val_chemicals=None, val_proteins=None, val_labels=None,
              sample_weights_by_epoch=None):
        if sample_weights_by_epoch is None:
            sample_weights_by_epoch = create_uniform_weights(len(train_chemicals), self.n_epochs)

        train_chemical_vectors = self.vectorize_chemicals(train_chemicals)
        train_protein_vectors = self.vectorize_proteins(train_proteins)
        train_labels = np.array(train_labels)

        val_tuple = None
        if val_chemicals is not None and val_proteins is not None and val_labels is not None:
            val_chemical_vectors = self.vectorize_chemicals(val_chemicals)
            val_protein_vectors = self.vectorize_proteins(val_proteins)
            val_tuple = ([val_chemical_vectors, val_protein_vectors], np.array(val_labels))

        train_stats_over_epochs = {'mse': [], 'rmse': [], 'r2': []}
        val_stats_over_epochs = train_stats_over_epochs.copy()
        for e in range(self.n_epochs):
            self.model.fit(x=[train_chemical_vectors, train_protein_vectors],
                           y=train_labels,
                           sample_weight=sample_weights_by_epoch[e],
                           validation_data=val_tuple,
                           batch_size=self.batch_size,
                           epochs=1)

            train_stats = evaluate_predictions(y_true=train_labels,
                                               y_preds=self.predict(train_chemicals, train_proteins),
                                               metrics=list(train_stats_over_epochs.keys()))
            for metric, stat in train_stats.items():
                train_stats_over_epochs[metric].append(stat)

            if val_tuple is not None:
                val_stats = evaluate_predictions(y_true=val_labels,
                                                 y_preds=self.predict(val_tuple[0], val_tuple[1]),
                                                 metrics=list(val_stats_over_epochs.keys()))
                for metric, stat in val_stats.items():
                    val_stats_over_epochs[metric].append(stat)

        self.history['train'] = train_stats_over_epochs
        if val_stats_over_epochs is not None:
            self.history['val'] = val_stats_over_epochs

        return self.history

    def predict(self, chemicals, proteins):
        chemical_vectors = self.vectorize_chemicals(chemicals)
        protein_vectors = self.vectorize_proteins(proteins)
        return self.model.predict([chemical_vectors, protein_vectors]).tolist()

    def save(self, path):
        path = strip_path(path)
        # print('Saving the model')
        self.model.save(f'{path}/model')

        with open(f'{path}/history.json', 'w') as f:
            json.dump(self.history, f, indent=4)

        donot_copy = {'model', 'history'}
        dct = {k: v for k, v in self.__dict__.items() if k not in donot_copy}
        with open(f'{path}/params.json', 'w') as f:
            json.dump(dct, f, indent=4)
