
# JUST-VERDICT
**Author:** Prahlad Koratamaddi
**UNI:** pk2743

JUST-VERDICT is a novel unsupervised framework for Justification Production in Fact-Checking using Knowledge Graphs and Data-To-Text Generation. This framework was developed as part of the Natural Language Generation and Summarization course by Prof. Kathleen McKeown at Columbia University.

#### Disclaimer
This repository does not contain certain important bulky folders, such as `data` and `llms`, which are used in the notebooks and code. The data folder is available for all LionMail users on the link here: [data](https://drive.google.com/drive/folders/1pHBCYt1SV1Rswj_aaG7rGbHk1Pkn22V1?usp=sharing). When setting this repo, please create the `llms` folder and store a `.gguf` file of a LLaMA-7B model. For my project, I used the [llama-2-7b.Q4_K_M.gguf](https://huggingface.co/TheBloke/Llama-2-7B-GGUF/blob/main/llama-2-7b.Q4_K_M.gguf) model from [Llama-2-7B-GGUF on HuggingFace](https://huggingface.co/TheBloke/Llama-2-7B-GGUF).

## Core Code Paths
1) **Data downloading, generation, preprocessing**: 
I conduct exploratory data analysis of the baseline PUBHEALTH dataset in the notebook `Explore PUBHEALTH Dataset.ipynb`. 
The data was available to download from this repository: [Health Fact Checking](https://github.com/neemakot/Health-Fact-Checking)
The code for Knowledge Graph Creation for all approaches (REBEL, FRED, SpaCy, LLaMa 7B, ChatGPT-3.5) is shown in the `Creating RDF Triples for PUBHEALTH.ipynb` notebook. Each data source is grouped under a markdown cell for convenience. REBEL, FRED, SpaCy and LLaMa-7B were used over the entire PUBHEALTH dataset, however ChatGPT-3.5 was used for a small randomized sample of 50 claims and evidences.

2) **Training Baselines and Experiments**:
The `Process Created RDF Triples.ipynb` notebook consists of code for Redundancy Removal, Evidence Retrieval, and Entailment Scoring steps for all 5 data sources. In addition to varying data sources, the code also explores ablation studies to evaluate individual components in JUST-VERDICT for the REBEL model, by maintaining versions with and without Redundancy Removal, and with randomized Evidence Retrieval (i.e., instead of pairing the most relevant evidence triple with a claim, a random evidence triple is paired instead).
The [PUBHEALTH paper](https://aclanthology.org/2020.emnlp-main.623/) baseline implementations were done by cloning [SciBERT](https://github.com/allenai/scibert),  [PreSumm](https://github.com/nlpyang/PreSumm), and the main [PUBHEALTH](https://github.com/neemakot/Health-Fact-Checking) repositories as is.

3) **Main Model Implementations**:
The `Process Created RDF Triples.ipynb` notebook uses the RoBERTa sentence-level semantic similarity model and the DeBERTa NLI textual entailment scoring models. These models are implemented in the `semantic_roberta.py` and `textual_entailment.py` Python code files respectively. The library powering the FRED knowledge graph creation is in the `fredlib.py` file.

4) **Evaluating Model Output**: 
The "Verdict Prediction" and "Justification Production" notebooks contain the code that evaluates all 5 data sources and ablations on REBEL as well. The results can be found in the `results/` folder in this repo. In addition to measuring verdict classification scores and justification fluency, faithfulness scores, I also present observations on the mean time to respond (MTTR) broken down into individual stages.

5) **Demo for End-to-End Fact Checking**:
The Python file `demo_end_to_end_fact_checking` consists of the code to demonstrate JUST-VERDICT in action. All of the dependant code had to be optimized for garbage collection because of memory constraints in loading multiple models at once, therefore onloading and offloading the models from RAM had to be done. These dependent modules are present under the `helpers/` folder. 

## The Basics
### What is this project about?
This project aims to automate two key tasks in fact-checking, verdict prediction and justification production, with the latter being the main value add. Given a textual claim and evidence as input, the models in this project provide the verdict (supports, contradicts, or not enough information), as well as the justification revealing the model's reasoning on how it arrived at the verdict.

### What is the approach?
Most automated fact-checking research requires additional finetuning on particular types of data, making them domain-specific. To achieve this in an efficient and universal way, models used in JUST-VERDICT are used out of the box without any additional finetuning or training, with few-shot prompting which remains the same irrespective of the domain of claims and evidences.
At a high level, JUST VERDICT consists of the following steps:
1. **Knowledge Graph Creation:** Both the claim and evidence are converted to knowledge graphs. 
2. **Redundancy Removal:** Any redundant or semantically similar edges are dropped to optimize speed and memory for next steps. 
3. **Evidence Retrieval:** For each edge (a triple) in the claim graph, fetches the most semantically similar evidence graph edge. 
4. **Entailment Scoring:** Does the claim (hypothesis) entail from the premise (most related evidence)? This scoring identifies that, and assigns a score to each claim edge. 
5. **Verdict Prediction:** Aggregating entailment scores of the claim graph, the verdict is predicted. 
6. **Justification Production:** Using a Data to Text generation model to convert a pair of claim and evidence triples with their verdict into a textual justification statement.

### What dataset is being used to evaluate JUST-VERDICT?
The [PUBHEALTH](https://github.com/neemakot/Health-Fact-Checking) dataset is used, because, among other reasons, it consists of a claim, evidence, verdict label as well as human-annotated reference justifications. 