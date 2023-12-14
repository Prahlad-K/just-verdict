from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
from sentence_transformers import CrossEncoder
import random

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_topk_similar_evidences(claim, evidences, k=1):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained('jhlee3421/faq-semantic-klue-roberta-large')
    model = AutoModel.from_pretrained('jhlee3421/faq-semantic-klue-roberta-large').to(device)

    if k > len(evidences):
        return evidences
    sentences = [claim] + evidences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(device)
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        sentence_embeddings = sentence_embeddings.detach().cpu().numpy()
    
    sim_scores = cosine_similarity([sentence_embeddings[0]], sentence_embeddings[1:])[0]

    topk_evidence_indices = np.argpartition(sim_scores, -1*k)[-1*k:]

    topk_evidences = [evidences[i] for i in topk_evidence_indices]
    
    del tokenizer
    del model
    torch.cuda.empty_cache()

    return topk_evidences


def get_entailment(evidence_claim_pairs):
    
    model = CrossEncoder('cross-encoder/nli-deberta-v3-large')

    scores = model.predict(evidence_claim_pairs)
    normalized_scores = scores / np.linalg.norm(scores, axis=1, keepdims=True)
    label_mapping = ['contradiction', 'entailment', 'neutral']
    # labels = [label_mapping[score_max] for score_max in scores.argmax(axis=1)]

    del model
    del scores

    return normalized_scores

def pair_most_related_evidence(claim_triples, evidence_triples, k, randomize=False):

    relevant_claim_evidence_pairs = {}
    for claim_triple in claim_triples:
        if randomize:
            topk_evidences = random.sample(evidence_triples, k)
        else:
            topk_evidences = get_topk_similar_evidences(claim_triple, evidence_triples, k)
        relevant_claim_evidence_pairs[claim_triple] = topk_evidences

    return relevant_claim_evidence_pairs


def get_entailment_scores(claim_evidence_pairs):
    
    evidence_claim_tuples = []
    for claim, evidences in claim_evidence_pairs.items():
        evidence_claim_tuples.append((evidences[0], claim))

    return get_entailment(evidence_claim_tuples)