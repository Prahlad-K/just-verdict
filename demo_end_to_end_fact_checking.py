import json
import time
import sys
sys.path.append('./helpers/')

import warnings
warnings.filterwarnings("ignore")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from helpers.kg_creation_rebel import from_text_to_kb
from helpers.kg_creation_fred import capture_FRED_kg  
from helpers.kg_creation_spacy import  generate_spacy_kgs 
from helpers.kg_creation_llama import  generate_kg_from_llama 
from helpers.redundancy_removal import get_cleaned_triples
from helpers.evidence_retrieval import pair_most_related_evidence, get_entailment_scores
from helpers.verdict_prediction import get_verdict
from helpers.justification_production import get_justification


VALID_LABELS = {'true', 'false', 'mixture', 'unproven'}

def get_valid_input(prompt):
    while True:
        user_input = input(prompt).strip()
        if user_input:
            return user_input
        else:
            print("Please enter a non-empty string.")

def get_default_or_user_input(prompt, default_value):
    user_choice = input(f"{prompt} (Press 'Y' or 'y' for default, otherwise enter your own): ").strip().lower()
    if user_choice == 'y' or user_choice == '':
        return default_value
    else:
        return get_valid_input(f"Enter {prompt.lower()}:\n")

# Default inputs
default_claim = "The earth is flat."
default_evidence = "The earth is round according to science."
default_label_actual = "false"
default_justification_actual = "The earth is not flat, because according to science it is round."

# Get user inputs or use default values
claim = get_default_or_user_input("Claim", default_claim)
evidence = get_default_or_user_input("Evidence", default_evidence)

while True:
    label_actual = get_default_or_user_input("Label Actual (true, false, mixture, unproven)", default_label_actual).lower()
    if label_actual in VALID_LABELS:
        break
    else:
        print("Invalid label. Please enter true, false, mixture, or unproven.")

justification_actual = get_default_or_user_input("Justification Actual", default_justification_actual)

print()

# Now you have valid input in the variables: claim, evidence, label_actual, and justification_actual
print("Valid input received:")
print(f"Claim: {claim}")
print(f"Evidence: {evidence}")
print(f"Label Actual: {label_actual}")
print(f"Justification Actual: {justification_actual}")

config = None
with open('./config.json', "r") as f:
    config = json.load(f)

if not config:
    print('Config file was not read correctly, please retry.')
    sys.exit(1)

print()
print('Please standby, processing.......')
print()

backup_claim_kg = from_text_to_kb(claim)
backup_evidence_kg = from_text_to_kb(evidence)

begin = time.time()

if config['kg_creation'] == 'REBEL':
    claim_kg = backup_claim_kg
    evidence_kg = backup_evidence_kg

elif config['kg_creation'] == 'FRED':
    claim_kg = capture_FRED_kg(claim, "./data/fred/examples/claim.rdf")
    # Throttle limits!
    time.sleep(12)
    evidence_kg = capture_FRED_kg(evidence, "./data/fred/examples/evidence.rdf")
    if claim_kg is None or len(claim_kg)==0:
        claim_kg = backup_claim_kg
    if evidence_kg is None or len(evidence_kg)==0:
        evidence_kg = backup_evidence_kg

elif config['kg_creation'] == 'SPACY':
    claim_kg = generate_spacy_kgs(claim)
    evidence_kg = generate_spacy_kgs(evidence)
    if claim_kg is None or len(claim_kg)==0:
        claim_kg = backup_claim_kg
    if evidence_kg is None or len(evidence_kg)==0:
        evidence_kg = backup_evidence_kg

elif config['kg_creation'] == 'LLAMA':
    claim_kg = generate_kg_from_llama(claim)
    evidence_kg = generate_kg_from_llama(evidence)
    if claim_kg is None or len(claim_kg)==0:
        claim_kg = backup_claim_kg
    if evidence_kg is None or len(evidence_kg)==0:
        evidence_kg = backup_evidence_kg

else:
    print('Incorrect specification for config kg_creation, can only be one of REBEL, FRED, SPACY or LLAMA. Please re-try.')
    sys.exit(1)

print('Created claim and evidence knowledge graphs......')

if config['avoid_redundancy_removal']:
    cleaned_claim_triples = get_cleaned_triples(claim_kg, clean=False)
    cleaned_evidence_triples = get_cleaned_triples(evidence_kg,clean=False)

    print('Skipped cleaning both knowledge graphs, outputs may sound redundant......')
else:
    cleaned_claim_triples = get_cleaned_triples(claim_kg)
    cleaned_evidence_triples = get_cleaned_triples(evidence_kg)

    print('Cleaned both knowledge graphs to avoid redundancies......')

if config['randomize_evidence_retrieval']:
    k = config['top_k_evidences']
    claim_evidence_pairs = pair_most_related_evidence(cleaned_claim_triples, cleaned_evidence_triples, 1, randomize=True)
    entailment_scores = get_entailment_scores(claim_evidence_pairs)

    print('Randomized evidence retrieval and completed entailment scoring......')

else:
    k = config['top_k_evidences']
    claim_evidence_pairs = pair_most_related_evidence(cleaned_claim_triples, cleaned_evidence_triples, 1)
    entailment_scores = get_entailment_scores(claim_evidence_pairs)

    print('Completed evidence retrieval and entailment scoring......')

verdict_actual = 1 if label_actual=='true' else (-1 if label_actual=='false' else 0)
verdict_predicted = get_verdict(entailment_scores)

justification_produced = get_justification(claim_evidence_pairs, entailment_scores)
end = time.time()

print()

print("Actual Verdict: ", "Not Enough Information" if verdict_actual==0 else ("Supports" if verdict_actual==1 else "Contradicts"))
print("Predicted Verdict: ", "Not Enough Information" if verdict_predicted==0 else ("Supports" if verdict_actual==1 else "Contradicts"))

print()

print("Actual Justification: ", justification_actual)
print("Produced Justification: ", justification_produced)

print()

print(f"Time Taken to Fact Check: {end-begin:3f} seconds.")
