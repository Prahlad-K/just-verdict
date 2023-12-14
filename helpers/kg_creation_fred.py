from fredlib import preprocessText, getFredGraph, openFredGraph


def clean_ontogolical_string(ontological_string):
    if '#' in ontological_string:
        if '_' in ontological_string:
            return ontological_string[ontological_string.find('#')+1:ontological_string.find('_')]
        else:
            return ontological_string[ontological_string.find('#')+1:]
    else:
        ontological_string =  ontological_string.split('/')[-1]
        if '_' in ontological_string:
            return ontological_string[:ontological_string.find('_')]
        else:
            return ontological_string

def get_RDF_triples_from_graph(edges):
    rdf_triples = []
    for triple in edges:
       
        subject = clean_ontogolical_string(triple[0]).lower()
        predicate = clean_ontogolical_string(triple[1])
        object = clean_ontogolical_string(triple[2]).lower()

        if subject==object:
            continue
    
        rdf_triple = {'head':subject, 'type':predicate, 'tail':object}
        rdf_triples.append(rdf_triple)
        
        # This is an invalid graph
        if 'runtime error' in object:
            return None

    return rdf_triples

def capture_FRED_kg(sentence, path, key = "fa042f72-771c-3d1b-9dfc-c84eee277cde"):
    g = getFredGraph(preprocessText(sentence), key, path)
    return get_RDF_triples_from_graph(g.getInfoEdges())