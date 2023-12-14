from transformers import MvpTokenizer, MvpForConditionalGeneration
import torch

def get_justification_statement(claim, evidence, verdict):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    tokenizer = MvpTokenizer.from_pretrained("RUCAIBox/mvp")
    model_with_mtl = MvpForConditionalGeneration.from_pretrained("RUCAIBox/mtl-data-to-text").to(device)

    claim_parts = claim[1:-1].split(', ')
    evidence_parts = evidence[1:-1].split(', ')

    if verdict == -1:
        claim_parts[1] = 'NOT ' + claim_parts[1].upper()
    else:
        claim_parts[1] = claim_parts[1].upper()
        
    evidence_parts[1] = evidence_parts[1].upper()
    formatted_claim = ' | '.join(claim_parts)
    formatted_evidence = ' | '.join(evidence_parts)

    verdict_text = ' [SEP] because '

    if verdict == 0:
        verdict_text = ' [SEP] unrelated to '

    prompt_text = "Describe the following data: " + formatted_claim + verdict_text + formatted_evidence
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
    ).to(device)

    generated_ids = model_with_mtl.generate(**inputs, max_length=64)
    generated_answer = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    del tokenizer
    del model_with_mtl
    del generated_ids
    del inputs

    torch.cuda.empty_cache()

    return generated_answer[0]


def get_justification(claim_evidences, entailment_scores):

    claims = list(claim_evidences.keys())

    index_values = [-1, 1, 0]

    justifications = []

    for i in range(len(claims)):
        which_verdict = entailment_scores[i].argmax()
        evidence = claim_evidences[claims[i]][0]
        claim = claims[i]
  
        justification = get_justification_statement(claim, evidence, index_values[which_verdict])

        justifications.append(justification)

    justification_mtl = ' '.join(justifications)
    
    return justification_mtl
