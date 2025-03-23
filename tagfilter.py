# Obtiene todos los "tags" del input del usuario
import openai
import spacy
from bson import ObjectId
from nltk.corpus import wordnet

from mongo_config import MongoDB
from openai_config import setup_openai
from serviceselection import selectServiceByIntent, serviceSelection, impServiceSelection

# Obtener la base de datos
db = MongoDB()

model_engine = setup_openai()

def getTagsFromService(service_id, domain):
    tags = []
    services = db.get_collection(domain, 'services')
    # Busco el servicio por id
    document = services.find_one({"_id": ObjectId(service_id)})

    # Check if the document exists and contains 'tags'
    if document and 'tags' in document:
        # Initialize example values
        tags = document['tags'][0]['name'].split(', ')

    #quito los tags repetidos
    unique_tags = list(set(tags))

    return unique_tags

def generateQuestionChatGPT(tags, domain):
    # Initialize an empty dictionary
    data_dict = {}

    for tag in tags:
        # Convert the ObjectId to a string before concatenating
        #prompt = "Can you give me an informal question that can be answered with yes or no to understand someone\'s preference regarding a parameter with this tag when selecting a restaurant?: \"" + str(tag) + "\""

        messages = [
            {
                "role": "user",
                "content": "Provide an informal question that can be answered with yes or no to understand someone's preference regarding a parameter with this tag when selecting a \"" + domain +": " +str(tag) + "\""
            }
        ]

        # Crear la solicitud de ChatCompletion
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # Puedes usar "gpt-4" si tienes acceso
            messages=messages,
            temperature=0.3,
            max_tokens=64,
            top_p=1,
            frequency_penalty=0.5,
            presence_penalty=0
        )

        # Extraer la respuesta generada por el modelo
        generated_text = response.choices[0].message.content
        print(generated_text)

        # Insert into the dictionary the tag and the question
        data_dict[tag] = generated_text

    print("Todos los datos necesarios:")
    print(data_dict)
    return data_dict

#Devolver los servicios filtrados según los tags que contengan
def filterServicesByTag(intentServices, userTags, domain):
    #tagServices = []
    services = {}
    services_bbdd = db.get_collection(domain, 'services')

    for service_id in intentServices:
        #Busco el servicio por id
        document = services_bbdd.find_one({"_id": ObjectId(service_id)})

        #Encuentro el servicio (debería siempre darlo ya que lo hemos guardado previamente)
        if document:
            # Itero el JSON y saco los intents que tiene definido el servicio
            for tag_document in document.get('tags', []):
                tags = tag_document.get("name", "")

                #divido en tokens
                tagList = {substring.strip() for substring in tags.split(',')}

                #Por cada etiqueta del servicio que esté en las etiquetas del usuario
                for tag in userTags:
                    if tag.lower() in tagList:
                        services[service_id] = services.get(service_id, 0) + 1

            #No hemos registrado ninguna etiqueta para ese servicio así que 0
            if service_id not in services:
                services[service_id] = 0

    # Ordena el diccionario por sus valores en orden ascendente
    sorted_services = dict(sorted(services.items(), key=lambda item: item[1]))
    return sorted_services

def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return synonyms

def getTags(input):
    tags = []
    synonyms = []
    synonyms_ = []

    #utilizo spacy para el procesamiento de lenguaje natural
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(input)

    for token in doc:
        #cogeré los adjetivos
        if (token.pos_ == 'ADJ') or (token.pos_ == 'NOUN'):
            tags.append(token.text)

    for tag in tags:
        synonyms_ = []
        synonyms_ = get_synonyms(tag)
        for synonym in synonyms_:
            synonyms.append(synonym)

    return synonyms

def tagFilter(userInput, intent, data_from_client, domain):
    # Saco los tags del input del usuario
    tags = getTags(userInput)

    # Elimino items repetidos
    unique_tags = list(set(tags))

    # Busco todos los servicios que tengan ese intent
    intentServices = selectServiceByIntent(intent, domain)
    print("intent service")
    print(intentServices)

    # Busco en los servicios si hay alguno con los tags, sino, los cojo todos.
    tagServices = filterServicesByTag(intentServices, unique_tags, domain)
    print("FILTRO POR TAG")
    print(tagServices)

    # Selecciono según los tags
    services = impServiceSelection(tagServices, userInput, data_from_client["filledslots"], intent, domain)
    print("FILTRO POR CAMPOS OBLIGATORIOS")
    print(services)

    return services

def getAditionalQuestions(services, userInput, intent, data_from_client, domain):
    # Cogeré los tags que no se repitan de los servicios:
    service_tags = set()
    first_iteration = True
    # Initialize an empty dictionary
    service_tags_dict = {}

    for service_id in services:
        # Cojo los tags de cada servicio
        tags = getTagsFromService(service_id, domain)
        if first_iteration:
            # For the first service, directly assign its tags to service_tags
            service_tags = set(tags)
            first_iteration = False
        else:
            # For the subsequent services, calculate the symmetric difference
            service_tags = service_tags.symmetric_difference(tags)

        # Store the service_id and the tags in the dictionary
        service_tags_dict[service_id] = service_tags

    print("SERVICE TAGS")
    print(service_tags)
    # Con estos tags generaré preguntas que me ayuden a discriminar entre los servicios
    ##########################################
    aditional_questions = generateQuestionChatGPT(service_tags, domain)
    print("ADDITIONAL QUESTIONS")
    print(aditional_questions)

    filledParams = data_from_client["filledslots"]
    for tag in service_tags:
        filledParams[tag] = ""

    # Formateo un poco las preguntas
    for key in aditional_questions:
        aditional_questions[key] = aditional_questions[key].replace('\n', '')  # This will remove newline characters
        aditional_questions[key] = aditional_questions[key].replace('"', '')  # This will remove double quotes

    return aditional_questions, filledParams