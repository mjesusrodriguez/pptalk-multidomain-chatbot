import spacy
from pymongo import MongoClient
from itertools import chain

# Cargar el modelo de SpaCy para inglés
nlp = spacy.load("en_core_web_md")

# Conexión a MongoDB
client = MongoClient("mongodb://localhost:27017/")

# Función para obtener los dos parámetros con la mayor combinación de frecuencias
def get_top_parameters_combined(domain):
    db = client[domain]  # Acceder a la base de datos correcta según el dominio
    parameter_collection = db["slot_ranking"]

    # Encontrar todos los parámetros y sus frecuencias
    all_parameters = parameter_collection.find()

    # Crear una lista de parámetros combinando ambas frecuencias
    combined_parameters = []
    for param in all_parameters:
        combined_frequency = param.get("service_frequency", 0) + param.get("user_frequency", 0)
        combined_parameters.append({
            "parameter": param["parameter"],
            "combined_frequency": combined_frequency,
            "values": param["values"]
        })

    # Ordenar por la frecuencia combinada en orden descendente
    combined_parameters.sort(key=lambda x: x["combined_frequency"], reverse=True)

    # Retornar los dos parámetros con mayor frecuencia combinada
    return combined_parameters[:2]

# Función para actualizar la frecuencia de los slots mencionados por el usuario, pero solo los solicitados
def update_frequencies_for_requested_slots(slots_list, reqSlots, domain):
    db = client[domain]
    parameter_collection = db["slot_ranking"]

    # Recorrer los slots que están en formato de lista de diccionarios
    for slots_dict in slots_list:
        # Verificar que cada elemento de la lista es un diccionario
        if isinstance(slots_dict, dict):
            # Para cada slot solicitado (reqSlots), comprobar si está en el diccionario actual
            for slot in reqSlots:
                # Verificar si el slot está presente y no es "Null"
                if slots_dict.get(slot) and slots_dict[slot].lower() != "null":
                    print(f"El usuario mencionó el parámetro {slot} con valor {slots_dict[slot]}")
                    # Incrementar la frecuencia de uso en la base de datos
                    parameter_collection.update_one(
                        {"parameter": slot},
                        {"$inc": {"user_frequency": 1}},  # Incrementar la frecuencia de uso del parámetro
                        upsert=True  # Crear el documento si no existe
                    )
                else:
                    print(f"El parámetro {slot} tiene el valor 'Null' o no fue mencionado, no se actualiza la frecuencia.")

# Función para generar ngrams
def generate_ngrams(tokens, n):
    return zip(*[tokens[i:] for i in range(n)])

def detect_and_update_other_slots(user_input, top_slots_list, domain):
    db = client[domain]
    parameter_collection = db["slot_ranking"]

    # Convertir los top slots en un conjunto para fácil comparación
    top_slots = set(slot['parameter'] for slot in top_slots_list)

    # Tokenizar el input del usuario usando SpaCy
    user_input_doc = nlp(user_input.lower())  # Convertimos el input a minúsculas para evitar problemas de coincidencia

    # Extraer las palabras del input tokenizado
    user_tokens = [token.text for token in user_input_doc]

    # Generar ngrams del input del usuario (unigrams, bigrams, trigrams, etc.)
    unigrams = user_tokens
    bigrams = [' '.join(gram) for gram in generate_ngrams(user_tokens, 2)]
    trigrams = [' '.join(gram) for gram in generate_ngrams(user_tokens, 3)]
    all_ngrams = list(chain(unigrams, bigrams, trigrams))

    # Buscar menciones de otros parámetros en la entrada del usuario
    all_parameters = parameter_collection.find()

    for param in all_parameters:
        param_name = param["parameter"]
        # Verificar si este parámetro está fuera de los slots solicitados
        if param_name not in top_slots:
            # Buscar coincidencias directas en los valores del parámetro
            for value in param["values"]:
                value_lower = value.lower()  # Convertimos el valor a minúsculas para coincidencia insensible a mayúsculas
                if value_lower in all_ngrams:
                    print(f"El usuario mencionó el parámetro {param_name} relacionado con {value}, actualizando frecuencia.")
                    parameter_collection.update_one(
                        {"parameter": param_name},
                        {"$inc": {"user_frequency": 1}},
                        upsert=True
                    )
                else:
                    print(f"No se encontró coincidencia para {value_lower} en el input del usuario.")