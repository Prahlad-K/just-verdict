import json
import time
import sys
sys.path.append('./helpers/')

from helpers.kg_creation_rebel import from_text_to_kb
from helpers.kg_creation_fred import capture_FRED_kg  
from helpers.kg_creation_spacy import  generate_spacy_kgs 
from helpers.kg_creation_llama import  generate_kg_from_llama 
from helpers.redundancy_removal import get_cleaned_triples
from helpers.evidence_retrieval import pair_most_related_evidence, get_entailment_scores
from helpers.verdict_prediction import get_verdict
from helpers.justification_production import get_justification


claim = "The earth is flat."
evidence = "The earth is round according to science."
label_actual = "false"
justification_actual = "The earth is not flat, because according to science it is round."

config = None
with open('config.json', "r") as f:
    config = json.load(f)

if not config:
    print('Config file was not read correctly, please retry.')
    sys.exit(1)

begin = time.time()

if config['kg_creation'] == 'REBEL':
    claim_kg = from_text_to_kb(claim)
    evidence_kg = from_text_to_kb(evidence)

elif config['kg_creation'] == 'FRED':
    claim_kg = capture_FRED_kg(claim, "./data/fred/examples/claim.rdf")
    # Throttle limits
    time.sleep(12)
    evidence_kg = capture_FRED_kg(evidence, "./data/fred/examples/evidence.rdf")

elif config['kg_creation'] == 'SPACY':
    claim_kg = generate_spacy_kgs(claim)
    evidence_kg = generate_spacy_kgs(evidence)

else:
    claim_kg = generate_kg_from_llama(claim)
    evidence_kg = generate_kg_from_llama(evidence)

cleaned_claim_triples = get_cleaned_triples(claim_kg)
cleaned_evidence_triples = get_cleaned_triples(evidence_kg)

claim_evidence_pairs = pair_most_related_evidence(cleaned_claim_triples, cleaned_evidence_triples, 1)
entailment_scores = get_entailment_scores(claim_evidence_pairs)

verdict_actual = 1 if label_actual=='true' else (-1 if label_actual=='false' else 0)
verdict_predicted = get_verdict(entailment_scores)

justification_produced = get_justification(claim_evidence_pairs, entailment_scores)
end = time.time()

print("Actual Verdict: ", "Not Enough Information" if verdict_actual==0 else ("Supports" if verdict_actual==1 else "Contradicts"))
print("Predicted Verdict: ", "Not Enough Information" if verdict_predicted==0 else ("Supports" if verdict_actual==1 else "Contradicts"))

print()

print("Actual Justification: ", justification_actual)
print("Produced Justification: ", justification_produced)

print()

print(f"Time Taken to Fact Check: {end-begin:3f} seconds.")
