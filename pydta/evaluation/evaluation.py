from itertools import combinations

from sklearn.metrics import mean_squared_error, r2_score


def ci(gold_truths, predictions):
    gold_combs, pred_combs = combinations(gold_truths, 2), combinations(predictions, 2)
    nominator, denominator = 0, 0
    for (g1, g2), (p1, p2) in zip(gold_combs, pred_combs):
        if g2 > g1:
            nominator = nominator + 1 * (p2 > p1) + 0.5 * (p2 == p1)
            denominator = denominator + 1

    return float(nominator / denominator)


def mse(gold_truths, predictions):
    return float(mean_squared_error(gold_truths, predictions, squared=True))


def rmse(gold_truths, predictions):
    return float(mean_squared_error(gold_truths, predictions, squared=False))


def r2(gold_truths, predictions):
    return float(r2_score(gold_truths, predictions))


def evaluate_predictions(y_true, y_preds, metrics):
    metrics = [metric.lower() for metric in metrics]
    name_to_fn = {'ci': ci, 'r2': r2, 'rmse': rmse, 'mse': mse}
    return {metric: name_to_fn[metric](y_true, y_preds) for metric in metrics}
