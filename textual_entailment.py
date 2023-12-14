# START: Inspired by https://huggingface.co/cross-encoder/nli-deberta-v3-large
from sentence_transformers import CrossEncoder
import numpy as np

model = CrossEncoder('cross-encoder/nli-deberta-v3-large')

def get_entailment(evidence_claim_pairs):
    scores = model.predict(evidence_claim_pairs)
    normalized_scores = scores / np.linalg.norm(scores, axis=1, keepdims=True)
    label_mapping = ['contradiction', 'entailment', 'neutral']
    labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]

    # print(labels)
    return normalized_scores
# END: Inspired by https://huggingface.co/cross-encoder/nli-deberta-v3-large