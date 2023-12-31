from transformers import MvpTokenizer, MvpForConditionalGeneration
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"

tokenizer = MvpTokenizer.from_pretrained("RUCAIBox/mvp")
model_with_mtl = MvpForConditionalGeneration.from_pretrained("RUCAIBox/mtl-data-to-text").to(device)


def get_justification_statement(claim, evidence, verdict):
    claim_parts = claim[1:-1].split(', ')

    if len(evidence) > 1:
        evidence_parts = evidence[1:-1].split(', ')
    else:
        evidence_parts = []
    
    if verdict == -1:
        claim_parts[1] = 'NOT ' + claim_parts[1].upper()
    else:
        claim_parts[1] = claim_parts[1].upper()
        
    formatted_claim = ' | '.join(claim_parts)

    if len(evidence_parts) > 1:
        evidence_parts[1] = evidence_parts[1].upper()
        formatted_evidence = ' | '.join(evidence_parts)
    else:
        formatted_evidence = ""

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

    return generated_answer[0]

# claim_evidence_verdicts = [('(A baby died at an unnamed medical facility, participant, its parents)', '(Confederate, owner of, Confederate flag)', -1), ('(Black Saturday, has effect, Black Lives Matter)', '(cardiac event, has effect, death)', -1)]

# get_justification_statement(claim_evidence_verdicts[0])
# get_justification_statement(claim_evidence_verdicts[1])



# claim2 = "(square, shape of, pizza)"
# evidence2 = "(pizza, is, round)"
# verdict2 = -1
# get_justification_statement(claim2, evidence2, verdict2)
