from llama_cpp import Llama


def generate_kg_from_llama(text):
      llama7b = Llama(model_path="./llms/llama-2-7b.Q5_K_M.gguf", n_ctx=2048, verbose=False)

      try:
            prompt = """
            Output RDF triples given an input text as follows:
            Input text: The earth is flat, and the sun rises in the east.
            Output RDF triples: [{'head':'earth', 'type':'shape', 'tail':'flat'}, {'head':'sun', 'type':'rises', 'tail':'east'}]
            Input text: San Fransisco is the capital of California, a state in the United States.
            Output RDF triples: [{'head':'San Fransisco', 'type':'capital', 'tail':'California'}, {'head':'California', 'type':'state', 'tail':'United States'}]
            Input text: """ + text + """
            Output RDF triples: 
            """

            output = llama7b(
                  prompt, # Prompt
                  max_tokens=None, # Generate an unlimited number of tokens potentially
                  stop=[']'], 
                  echo=False, # Echo the prompt back in the output
                  seed=42,
                  temperature=0
            ) 

            triples_string = output['choices'][0]['text'] + ']'
            triples = eval(triples_string)
            del llama7b
            return triples
      
      except Exception as e:
            del llama7b
            return None
      