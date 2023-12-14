
import spacy
nlp = spacy.load("en_core_web_sm")

def generate_spacy_kgs(sentence):
    triples = []
    doc = nlp(sentence)

    subject = None
    for token in doc:
        if "subj" in token.dep_:
            subject = token.text
        elif "obj" in token.dep_:
            obj = token.text
            predicate = [ancestor.text for ancestor in token.ancestors if ancestor.dep_ == "ROOT"][0]
            if subject:
                triples.append({'head':subject, 'type':predicate, 'tail':obj})

    return triples