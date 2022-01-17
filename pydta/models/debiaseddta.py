import random

import numpy as np

from ..utils import strip_path, create_uniform_weights


class DebiasedDTA:
    def __init__(self, guide_cls, predictor_cls, mini_val_frac=0.2, n_bootstrapping=10,
                 guide_params=None, predictor_params=None):

        self.guide_cls = guide_cls
        self.predictor_cls = predictor_cls
        self.mini_val_frac = mini_val_frac
        self.n_bootstrapping = n_bootstrapping
        # if weight_adaptation_strategy is not None:
        #     weight_adaptation_strategy = weight_adaptation_strategy.lower()
        #     if weight_adaptation_strategy not in {'bd', 'bg'}:
        #         raise ValueError('Invalid weight adaptation strategy')

        # self.weight_adaptation_strategy = weight_adaptation_strategy

        self.guide_params = {} if guide_params is None else guide_params
        self.predictor_params = {} if predictor_params is None else predictor_params
        self.predictor_instance = self.predictor_cls(
            **self.predictor_params)
        if 'n_epochs' not in self.predictor_instance.__dict__:
            raise ValueError(
                'Strong learner must have a field named "n_epochs" to be debiased')

    @staticmethod
    def save_importance_coefficients(interactions, importance_coefficients, savedir):
        savedir = strip_path(savedir)
        dump_content = []
        for interaction_id, chemical, protein, label in interactions:
            importance_coefficient = importance_coefficients[interaction_id]
            dump_content.append(
                f'{chemical},{protein},{label},{importance_coefficient}')
        dump = '\n'.join(dump_content)
        with open(savedir) as f:
            f.write(dump)

    def learn_importance_coefficients(self, train_chemicals, train_proteins, train_labels, savedir=None):
        train_size = len(train_chemicals)
        train_interactions = list(
            zip(range(train_size), train_chemicals, train_proteins, train_labels))
        mini_val_data_size = int(train_size * self.mini_val_frac) + 1
        interaction_id_to_sq_diff = [[] for _ in range(train_size)]

        for i in range(self.n_bootstrapping):
            random.shuffle(train_interactions)
            n_mini_val = int(1 / self.mini_val_frac)
            for mini_val_ix in range(n_mini_val):
                val_start_ix = mini_val_ix * mini_val_data_size
                val_end_ix = val_start_ix + mini_val_data_size
                mini_val_interactions = train_interactions[val_start_ix: val_end_ix]
                mini_train_interactions = train_interactions[:val_start_ix] + \
                    train_interactions[val_end_ix:]
                assert len(mini_train_interactions) + \
                    len(mini_val_interactions) == train_size

                mini_train_chemicals = [interaction[1]
                                        for interaction in mini_train_interactions]
                mini_train_proteins = [interaction[2]
                                       for interaction in mini_train_interactions]
                mini_train_labels = [interaction[3]
                                     for interaction in mini_train_interactions]
                guide_instance = self.guide_cls(
                    **self.guide_params)
                guide_instance.train(
                    mini_train_chemicals, mini_train_proteins, mini_train_labels)

                mini_val_chemicals = [interaction[1]
                                      for interaction in mini_val_interactions]
                mini_val_proteins = [interaction[2]
                                     for interaction in mini_val_interactions]
                preds = guide_instance.predict(
                    mini_val_chemicals, mini_val_proteins)
                mini_val_labels = [interaction[3]
                                   for interaction in mini_val_interactions]
                mini_val_sq_diffs = (
                    np.array(mini_val_labels) - np.array(preds)) ** 2
                mini_val_interaction_ids = [interaction[0]
                                            for interaction in mini_val_interactions]
                for interaction_id, sq_diff in zip(mini_val_interaction_ids, mini_val_sq_diffs):
                    interaction_id_to_sq_diff[interaction_id].append(sq_diff)

        for ix, sq_diffs in enumerate(interaction_id_to_sq_diff):
            assert len(sq_diffs) == self.n_bootstrapping

        interaction_id_to_med_diff = [
            np.median(diffs) for diffs in interaction_id_to_sq_diff]
        importance_coefficients = [
            med / sum(interaction_id_to_med_diff) for med in interaction_id_to_med_diff]

        if savedir is not None:
            DebiasedDTA.save_importance_coefficients(
                train_interactions, importance_coefficients, savedir)

        return importance_coefficients

    def train(self, train_chemicals, train_proteins, train_labels,
              val_chemicals=None, val_proteins=None, val_labels=None, coeffs_save_path=None):
        train_chemicals = train_chemicals.copy()
        train_proteins = train_proteins.copy()
        if len(train_chemicals) != len(train_proteins):
            raise ValueError(
                'The number of training chemicals and proteins are different')

        importance_coefficients = self.learn_importance_coefficients(train_chemicals,
                                                                     train_proteins,
                                                                     train_labels,
                                                                     savedir=coeffs_save_path)

        n_epochs = self.predictor_instance.n_epochs
        ic = np.array(importance_coefficients)
        # if self.weight_adaptation_strategy == 'bd':
        weights_by_epoch = [1 - (e / n_epochs) + ic * (e / n_epochs)
                            for e in range(n_epochs)]
        # elif self.weight_adaptation_strategy == 'bg':
        #     weights_by_epoch = [(e / n_epochs) + (b - b * (e / n_epochs))
        #                         for e in range(n_epochs)]
        # else:  # no strategy
        #     weights_by_epoch = create_uniform_weights(len(b), n_epochs)

        if val_chemicals is not None and val_proteins is not None and val_labels is not None:
            return self.predictor_instance.train(train_chemicals, train_proteins, train_labels,
                                                 val_chemicals=val_chemicals, val_proteins=val_proteins, val_labels=val_labels,
                                                 sample_weights_by_epoch=weights_by_epoch)

        return self.predictor_instance.train(train_chemicals, train_proteins, train_labels,
                                             sample_weights_by_epoch=weights_by_epoch)
