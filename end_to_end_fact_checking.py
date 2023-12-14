import json

import sys
sys.path.append('./helpers/')

from helpers.kg_creation_rebel import from_text_to_kb  
from helpers.redundancy_removal import get_cleaned_triples
from helpers.evidence_retrieval import pair_most_related_evidence, get_entailment_scores
from helpers.verdict_prediction import get_verdict
from helpers.justification_production import get_justification


claim = "The earth is flat."
evidence = "The earth is round according to science."
actual_label = "false"
reference_justification = "The earth is not flat, because according to science it is round."

config = None
with open('config.json', "r") as f:
    config = json.load(f)

if not config:
    print('Config file was not read correctly, please retry.')


claim_kg = from_text_to_kb(claim)
evidence_kg = from_text_to_kb(evidence)

print(claim_kg)
print(evidence_kg)

cleaned_claim_triples = get_cleaned_triples(claim_kg)
cleaned_evidence_triples = get_cleaned_triples(evidence_kg)

print(cleaned_claim_triples)
print(cleaned_evidence_triples)

claim_evidence_pairs = pair_most_related_evidence(cleaned_claim_triples, cleaned_evidence_triples, 1)
entailment_scores = get_entailment_scores(claim_evidence_pairs)

print(claim_evidence_pairs)
print(entailment_scores)

verdict_actual = 1 if actual_label=='true' else (-1 if actual_label=='false' else 0)
verdict_predicted = get_verdict(entailment_scores)

print(verdict_actual, verdict_predicted)

justification = get_justification(claim_evidence_pairs, entailment_scores)

print(reference_justification)
print(justification)