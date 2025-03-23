import openai
from openai_config import setup_openai
import json

model_engine = setup_openai()


def opendomainconversation(input, dialogue_history = None):
    # Convertir el historial de diálogo (lista) a una cadena JSON
    dialogue_history_str = "\n".join([f"User: {entry['user']}\nChatbot: {entry['chatbot']}" for entry in dialogue_history]) if dialogue_history else ""
    print("DIALOGUE HISTORY: ", dialogue_history_str)
    if(dialogue_history_str == []):
        messages = [
            {
                "role": "system",
                "content": "You are an open-domain chatbot that engages in friendly, casual conversations with the user."
            },
            {
                "role": "user",
                "content": f"The current conversation is as follows:\n{dialogue_history_str}\nContinue the conversation with this input: {input}"
            }
        ]
    else:
        messages = [
            {
                "role": "system",
                "content": "You are an open-domain chatbot that engages in friendly, casual conversations with the user."
            },
            {
                "role": "user",
                "content": f"Start a conversation with this input: {input}"
            }
        ]

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # Puedes usar "gpt-4" si tienes acceso
        messages=messages,
        temperature=0.7,
        max_tokens=128,
        top_p=1,
        frequency_penalty=0.5,
        presence_penalty=0
    )

    # Extraer la respuesta generada por el modelo
    generated_text = response.choices[0].message.content
    print(generated_text)

    return generated_text