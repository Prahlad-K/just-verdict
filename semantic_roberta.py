from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings for
# sentences = ["(earth is flat, said to be the same as, disk)",
# "(disk, said to be the same as, earth is flat)",
# "(flat, facet of, earth)",
# "(flat, said to be the same as, disk)",
# "(disk, said to be the same as, flat)"]

# Load model from HuggingFace Hub
device = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained('jhlee3421/faq-semantic-klue-roberta-large')
model = AutoModel.from_pretrained('jhlee3421/faq-semantic-klue-roberta-large').to(device)

# Tokenize sentences
# encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

# Compute token embeddings
# with torch.no_grad():
#     model_output = model(**encoded_input)

# Perform pooling. In this case, mean pooling.
# sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

# sentence_embeddings = sentence_embeddings.detach().numpy()
# print(cosine_similarity(sentence_embeddings, sentence_embeddings))


def get_similarity_scores_triples(triples):
    sentences = triples
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(device)
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        sentence_embeddings = sentence_embeddings.detach().cpu().numpy()

    return(cosine_similarity(sentence_embeddings, sentence_embeddings))
    
def get_topk_similar_evidences(claim, evidences, k=1):
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
    # print(sim_scores)
    # print(topk_evidence_indices)
    # print(topk_evidences)

    return topk_evidences