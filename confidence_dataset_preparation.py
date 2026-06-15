import json
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler

df = pd.read_json('datasets/Qwen 2.5 Judge Answer.json')

def compute_ece(confidences, correctness, n_bins=10):
    confidences = np.array(confidences)
    correctness = np.array(correctness)

    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i+1])
        if np.sum(mask) == 0:
            continue

        acc = np.mean(correctness[mask])
        conf = np.mean(confidences[mask])
        weight = np.sum(mask) / len(confidences)

        ece += weight * abs(acc - conf)

    return ece

internal_scores = df['p_internal']
consistency_scores = df['p_consistency']
semantic_scores = df['p_semantic']
selfeval_scores = df['p_selfeval']
correctness = df['correctness']

ece_internal = compute_ece(internal_scores, correctness)
ece_consistency = compute_ece(consistency_scores, correctness)
ece_semantic = compute_ece(semantic_scores, correctness)
ece_selfeval = compute_ece(selfeval_scores, correctness)

ECE = {
    "internal": ece_internal,
    "consistency": ece_consistency,
    "semantic": ece_semantic,
    "selfeval": ece_selfeval
}


### Each signal contributes based on how well it is calibrated (ECE)
def fused_confidence(signals, ECE, eps=1e-6):
    numerator = 0.0
    denominator = 0.0

    for k in signals:
        weight = 1.0 / (ECE[k] + eps)
        numerator += weight * signals[k]
        denominator += weight

    return numerator / denominator


signals = {
    "internal": internal_scores,
    "consistency": consistency_scores,
    "semantic": semantic_scores,
    "selfeval": selfeval_scores
}

final_C = fused_confidence(signals, ECE)


def verbalize(score):

    if score >= 0.90:
        return "I am certain"

    elif score >= 0.75:
        return "I am likely correct"

    elif score >= 0.60:
        return "I think this is correct"

    elif score >= 0.40:
        return "I am unsure"

    else:
        return "I may be incorrect"

df["confidence_score"] = fused_confidence(signals, ECE); df["confidence_score"].describe()

df["verbal_confidence"] = (
    df["confidence_score"]
    .apply(verbalize)
)

print(df.head())

df['verbal_confidence'].value_counts()

df[['question','answer','confidence_score','verbal_confidence']].to_csv('calibrated_QA_dataset.csv', index=False)

print("Saved calibrated dataset with confidence scores and verbalizations.")
