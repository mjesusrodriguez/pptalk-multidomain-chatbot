import openai
from openai_config import setup_openai

model_engine = setup_openai()

def domain_manager_gpt(input):
    messages = [
        {
            "role": "user",
            "content": f'You are a domain classifier in a dialogue system. Classify the following input: "{input}". The input might refer to more than one domain. Return all relevant domains from the following: "restaurants", "hotels", "attractions", or "out-of-domain" if the input does not fit in any domain. Return the domains as a comma-separated list of words, even if there is only one relevant domain.'        }
    ]

    # Crear la solicitud de ChatCompletion
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # Puedes usar "gpt-4" si tienes acceso
        messages=messages,
        temperature=0,
        max_tokens=64,
        top_p=1,
        frequency_penalty=0.5,
        presence_penalty=0
    )

    # Extraer la respuesta generada por el modelo
    generated_text = response.choices[0].message.content
    # Dividir la cadena por comas y quitar espacios innecesarios
    final_response = [domain.strip() for domain in generated_text.split(',')]
    print(f"Final response (as list): {final_response}")

    return final_response