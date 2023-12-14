import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_similarity_scores_triples(triples):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained('jhlee3421/faq-semantic-klue-roberta-large')
    model = AutoModel.from_pretrained('jhlee3421/faq-semantic-klue-roberta-large').to(device)

    sentences = triples
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt').to(device)
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
        sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

        sentence_embeddings = sentence_embeddings.detach().cpu().numpy()

    del tokenizer
    del model
    torch.cuda.empty_cache()

    return(cosine_similarity(sentence_embeddings, sentence_embeddings))


def get_cleaned_triples(kg, clean=True):
    triples = [f"({triple['head']}, {triple['type']}, {triple['tail']})" for triple in kg]

    if clean:
        n = len(triples)
        sim_scores = get_similarity_scores_triples(triples)

        to_drop = np.zeros(n, dtype=bool)

        for i in range(n):
            # Skip sentences that have already been marked for dropping
            if to_drop[i]:
                continue

            for j in range(i + 1, n):
                # If the similarity score between sentences i and j is above the threshold, mark one of them for dropping
                if sim_scores[i, j] >= 0.9:
                    to_drop[i] = True
                    break

        # Return the indices of sentences that are not marked for dropping
        clean_triples = [triples[i] for i in range(n) if not to_drop[i]]
        return clean_triples
    else:
        return triples

