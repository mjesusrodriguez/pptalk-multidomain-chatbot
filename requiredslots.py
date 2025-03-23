#Función que hace el slot filling sin tener el servicio (parametros obligatorios)
import json
import openai
from openai_config import setup_openai

model_engine = setup_openai()

def slotFillingRequired(slots, userinput):
    # Convert the slots list to a string
    slots_str = json.dumps(slots)
    prompt = "If I give you the prompt: \""+ userinput +"\", and the slots ["+ slots_str +"],  give me a JSON list with the slots and the values that are given in the prompt directly. If the value is not given, give the value \"Null\""
    print(prompt)

    # Generate a response from OpenAI based on the prompt
    response = openai.Completion.create(
        engine=model_engine,  # Choose the engine you prefer
        prompt=prompt,
        max_tokens=64,
        temperature=0.3
    )

    # Print the response from OpenAI
    print(response.choices[0].text.strip())
    return response.choices[0].text.strip()
