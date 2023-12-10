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

# evidence_claim_pairs = [("(round, shape of, earth)", "(flat, shape of, earth)"), ("(round, shape of, earth)", "(spherical, shape of, ball)")]
# evidence_claim_pairs2 = [("(pizza, shape, circle)", "(round, shape of, pizza)"), ("(night, lacks, sunlight)", "(night, has, moon)")]

# start = time.time()
# get_entailment(evidence_claim_pairs)
# end1 = time.time()
# get_entailment(evidence_claim_pairs2)
# end2 = time.time()

# print(end1-start)
# print(end2-end1)