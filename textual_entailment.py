from sentence_transformers import CrossEncoder
import numpy as np
import time

# model = CrossEncoder('cross-encoder/nli-MiniLM2-L6-H768')
model = CrossEncoder('cross-encoder/nli-deberta-v3-large')

# scores = model.predict()

#Convert scores to labels
# label_mapping = ['contradiction', 'entailment', 'neutral']
# labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]

# print(labels)

def get_entailment(evidence_claim_pairs):
    scores = model.predict(evidence_claim_pairs)
    normalized_scores = scores / np.linalg.norm(scores, axis=1, keepdims=True)
    label_mapping = ['contradiction', 'entailment', 'neutral']
    labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]

    # print(labels)
    return normalized_scores

# evidence_claim_pairs = [("Earth is round", "Sky is blue")]

# print(get_entailment(evidence_claim_pairs))