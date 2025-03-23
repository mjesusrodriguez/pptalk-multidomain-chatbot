import uuid

from bson import ObjectId
from flask import Flask, render_template, request, jsonify
import json
import random
import requests
import re

from disc_parameter import get_top_parameters_combined, update_frequencies_for_requested_slots, \
    detect_and_update_other_slots
from domain_manager import domain_manager_gpt
from intentrec import intentRecWithChatGPT
from mongo_config import MongoDB
from openai_config import setup_openai
from opendomain import opendomainconversation
from questionimprovement import improveQuestionchatGPT, createQuestionGPT
from questionretrieval import questionsRetrieval
from slotfilling import extractSlots, slotFillingGPT
from tagfilter import tagFilter, getAditionalQuestions

app = Flask(__name__)

# Obtener la base de datos
db = MongoDB()

#OpenAI Model
model_engine = setup_openai()

#Variable global para guardar el servicio
service_id = ""
#variable global para guardar el intent
intent = ""
# Inicializar el historial de la conversación
dialogue_history = {
    'useranswers': []
}
# Diccionario global para almacenar los datos por sesión
session_data = {}

#Devolver los servicios filtrados según los tags que contengan
def filterServicesByTag(intentServices, userTags, domain):
    #tagServices = []
    services = {}

    collection = db.get_collection(domain, 'services')

    for service_id in intentServices:
        #Busco el servicio por id
        document = collection.find_one({"_id": ObjectId(service_id)})

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

def detect_positive_answers(response_dict):
    positive_keywords = ["yes", "yeah", "yep", "sure", "absolutely", "definitely", "of course"]
    positive_tags = []

    for tag, answer in response_dict.items():
        response_lower = answer.lower()
        if any(word in response_lower for word in positive_keywords):
            positive_tags.append(tag)

    return positive_tags

# Lista de palabras clave de despedida
goodbye_keywords = ['goodbye', 'bye', 'see you', 'later', 'farewell', 'take care', 'thanks', 'thank you', 'talk to you later', 'bye bye']

# Función que detecta si se está despidiendo mediante patrones
def check_for_goodbye(user_input):
    # Normaliza el input del usuario a minúsculas y compara con las palabras clave
    for keyword in goodbye_keywords:
        if re.search(rf'\b{keyword}\b', user_input.lower()):
            return True
    return False

# Lista de frases comunes que indican un posible dominio abierto
open_domain_phrases = [
    r'what do you think',  # Detectar preguntas del tipo "What do you think..."
    r'tell me about',      # Preguntas más generales del tipo "Tell me about..."
    r'can you share',      # Preguntas como "Can you share..."
    r'what is your opinion',  # Preguntas como "What is your opinion..."
    r'explain to me',      # Expresiones como "Explain to me..."
    # Puedes añadir más patrones si es necesario
]

# Función para detectar dominio abierto mediante patrones
def detect_open_domain(user_input):
    for phrase in open_domain_phrases:
        if re.search(phrase, user_input.lower()):
            return True
    return False

def manage_open_dialogue(data):
    print("Entrando en diálogo de dominio abierto...")

    userAnswers = data.get('useranswers', [])
    userInput = data.get('userinput')

    # Lógica para manejar el diálogo abierto
    chatbot_answer = opendomainconversation(userInput, userAnswers)

    # Guardar la conversación
    userAnswers.append({"user": userInput, "chatbot": chatbot_answer})

    # Respuesta final al cliente, sin que el cliente controle el flujo
    return jsonify({
        'chatbot_answer': chatbot_answer,
        'useranswers': userAnswers,
        'dom': 'out-of-domain',
    }), 200

def service_selection(data):
    emptyParams = {}
    filledParams = {}

    # Debug: Print the received data
    print("Received data from manage_task_oriented:", data)

    tasks = data.get('tasks')  # Obtener el intent
    # Procesar el primer dominio-intent en el diccionario de tareas
    domain = data.get('domain')  # Obtener el primer dominio
    intent = data.get('intent')  # Obtener el intent de ese dominio
    print(f"Procesando dominio: {domain}, intent: {intent}")

    userInput = data.get('userinput')  # Obtener el input del usuario
    userAnswers = data.get('useranswers', [])  # Obtener el historial del diálogo

    reqSlots = data.get('reqslots')  # Obtener los slots requeridos

    if check_for_goodbye(userInput):
        return jsonify({"chatbot_answer": "Thank you for chatting! Goodbye!", "end_conversation": True})

    # Crear una lista para los parámetros dinámicos
    dynamic_params = reqSlots  # Todos los slots dinámicos

    # Selecciono un servicio
    services = tagFilter(userInput, intent, data, domain)

    # Voy a coger los parámetros discriminatorios de los servicios si son más de uno.
    if len(services) > 1:
        aditional_questions, filledParams = getAditionalQuestions(services, userInput, intent, data, domain)

        # Imprimir el tipo y contenido de cada elemento en services
        print("Tipo de elementos en services: ", [type(service) for service in services])
        print("Contenido de services: ", services)

        # Convertir cada elemento a cadena solo si es necesario
        services_as_strings = [str(service) for service in services]
        print("Servicios convertidos a cadena: ", services_as_strings)

        # Verificar que 'services' es una lista antes de proceder
        if not isinstance(services, list):
            print("Error: 'services' no es una lista, es de tipo:", type(services))
            return jsonify({"error": "Invalid format for services"}), 500

        # Convertir cada elemento de la lista a cadena de forma segura
        try:
            services_as_strings = [str(service) for service in services]
        except Exception as e:
            print(f"Error al convertir services a cadenas: {e}")
            return jsonify({"error": "Could not convert services to strings"}), 500

        return jsonify(
            {'questions': aditional_questions, 'filledslots': filledParams, 'intent': intent, 'userinput': userInput,
             'services': services_as_strings, 'useranswers': userAnswers, 'dom': domain,
             'reqslots': reqSlots, 'tasks': tasks}), 202
    else:
        # Selecciono el que se ha devuelto
        service_id = services[0]

        # Consulto en los servicios que tengo que campos se han rellenado ya y cuales faltan y devuelvo las preguntas.
        slots = extractSlots(intent, service_id, domain)
        slotFillingResponse = slotFillingGPT(userInput, slots, dialogue_history)

        # Verificar el tipo de respuesta
        print(f"Respuesta de slotFillingGPT: {slotFillingResponse}")

        # Convertir la respuesta en una lista de diccionarios
        sf_data = json.loads(slotFillingResponse)

        # Verifica si sf_data es una lista o un diccionario
        if isinstance(sf_data, list):
            # Iterar sobre cada elemento de la lista (donde cada elemento es un diccionario)
            for item in sf_data:
                if isinstance(item, dict):
                    for key, value in item.items():
                        if value is None:
                            emptyParams[key] = value
                        else:
                            filledParams[key] = value
        elif isinstance(sf_data, dict):
            # Si sf_data es un diccionario, itera sobre él directamente
            for key, value in sf_data.items():
                if value is None:
                    emptyParams[key] = value
                else:
                    filledParams[key] = value
        else:
            raise TypeError(f"Expected list or dict, got {type(sf_data)}")

        # Mostrar los resultados de los parámetros vacíos y llenos
        print(f"Parámetros vacíos (emptyParams): {emptyParams}")
        print(f"Parámetros llenos (filledParams): {filledParams}")

        # Asegúrate de que dynamic_params sea una lista o iterable
        if dynamic_params is None:
            dynamic_params = []  # Asignar una lista vacía si es None

        # Eliminar los slots que ya han sido llenados desde dynamic_params
        for param in dynamic_params:
            if param in emptyParams:
                emptyParams.pop(param)

        print("EMPTY PARAMS: ", emptyParams)

        # Evitar posibles errores, rellenar filledParams con los slots iniciales desde dynamic_params
        for param in dynamic_params:
            filledParams[param] = data.get("filledslots", {}).get(param, "")

        print("FILLED PARAMS: ", filledParams)

        # hago una llamada a la función que dado un intent y un id me da las preguntas.
        intent_info = questionsRetrieval(service_id, intent, domain)

        # Cuento la cantidad de parametros que hay en el json
        intent_info_json = intent_info[0].json
        slots = intent_info_json["intent"]["slots"]
        print("SLOTS", slots)

        json_slots = json.dumps(emptyParams)
        parsed_items = json.loads(json_slots)
        print("PARSED: ", parsed_items)

        # Guardo las preguntas de los parámetros que hacen falta.
        questions = {}

        for empty in parsed_items:
            if parsed_items[empty] is None:   # Solo procesar si el slot aún no ha sido llenado
                # Buscar en 'slots' el valor de la clave correspondiente a 'empty'
                if empty in slots:
                    # Eliminar comillas dobles si existen
                    question = slots[empty].replace('"', '')

                    # Mejorar la pregunta con el método
                    improved_question = improveQuestionchatGPT(question, domain)
                    questions[empty] = improved_question
                else:
                    print(f"Error: {empty} no se encontró en los slots.")

        # return questions
        print("QUESTIONS: ", questions)
        return jsonify(
            {'questions': questions, 'filledslots': filledParams, 'service_id': str(service_id), 'intent': intent, 'dom': domain,
             'useranswers': userAnswers, 'reqslots': reqSlots, 'tasks': tasks, 'final': True}), 202

def final_slot_filling(data):
    print("DEVUELVO DESDE SLOTFILLING:", data)
    emptyParams = {}
    filledParams = {}
    intent = data.get('intent')
    userInput = data.get('userinput')
    userAnswers = data.get('useranswers', [])
    tasks = data.get('tasks')
    reqslots = data.get('reqslots', [])

    # Procesar el primer dominio-intent en el diccionario de tareas
    domain = data.get('domain')  # Obtener el primer dominio
    intent = data.get('intent')  # Obtener el intent de ese dominio
    print(f"Procesando dominio: {domain}, intent: {intent}")

    # Obtener los slots dinámicos de reqslots
    reqSlots = data.get('reqslots', [])

    # Crear una lista para los parámetros dinámicos
    dynamic_params = reqSlots  # Todos los slots dinámicos

    if check_for_goodbye(userInput):
        return jsonify({"chatbot_answer": "Thank you for chatting! Goodbye!", "end_conversation": True})

    # cojo los datos de filledslots
    filledParams = data.get('filledslots', {})
    # Evalúo si para cada tag la respuesta es positiva o negativa.
    positive_tags = detect_positive_answers(filledParams)
    print("POSITIVE TAGS: ", positive_tags)

    # Filtro por esos tags con los servicios recogidos del cliente
    services = data.get('services', [])
    print("SERVICES: ", services)
    services = [ObjectId(service) for service in services]
    print("SERVICES: ", services)
    selected_services = []
    selected_services = filterServicesByTag(services, positive_tags, domain)
    print("SELECTED SERVICES BY NEW TAGS: ", selected_services)

    # Get the maximum value in the dictionary
    max_value = max(selected_services.values())

    # Get all keys (service_ids) with the maximum value
    max_value_services = [service_id for service_id, value in selected_services.items() if value == max_value]

    # Now you can check the length of max_value_services
    if len(max_value_services) > 1:
        # There are multiple services with the maximum value
        # You can select one of them randomly or based on some other criteria
        service_id = random.choice(max_value_services)
    else:
        # There is only one service with the maximum value
        service_id = max_value_services[0]

    print("SERVICE SELECTED: ", service_id)

    # CONTINUACIÓN DEL FLUJO NORMAL
    # extraigo los slots
    slots = extractSlots(intent, service_id, domain)

    # Le paso los slots a la tarea de SF, como es una tercera interacción, voy a pasarle el histórico del diálogo, para que rellene también
    # según ha contestado en las preguntas intermedias.
    slotFillingResponse = slotFillingGPT(userInput, slots, dialogue_history)
    print("Los slots rellenos con el cambio son:" + slotFillingResponse)

    # Convert the string to a dictionary
    sf_data = json.loads(slotFillingResponse)

    # Verificar si sf_data es una lista o un diccionario
    if isinstance(sf_data, list):
        # Si es una lista, iterar por la lista de diccionarios
        for item in sf_data:
            for key, value in item.items():
                if value == "Null":
                    emptyParams[key] = value
                else:
                    filledParams[key] = value
    elif isinstance(sf_data, dict):
        # Si es un diccionario, iterar directamente
        for key, value in sf_data.items():
            if value == "Null":
                emptyParams[key] = value
            else:
                filledParams[key] = value

    # Eliminar los slots que ya han sido llenados desde dynamic_params
    for param in dynamic_params:
        if param in emptyParams:
            emptyParams.pop(param)

    print("EMPTY PARAMS: ", emptyParams)

    # Evitar posibles errores, rellenar filledParams con los slots iniciales desde dynamic_params
    for param in dynamic_params:
        # Como reqSlots es una lista, verificamos si param está en esa lista
        if param in reqSlots:
            filledParams[param] = ""  # Puedes ajustar el valor predeterminado si es necesario
        else:
            filledParams[param] = "Null"  # O el valor por defecto si no está presente

    print("FILLED PARAMS: ", filledParams)

    # hago una llamada a la función que dado un intent y un id me da las preguntas.
    intent_info = questionsRetrieval(service_id, intent, domain)

    # Cuento la cantidad de parametros que hay en el json
    intent_info_json = intent_info[0].json
    slots = intent_info_json["intent"]["slots"]

    json_slots = json.dumps(emptyParams)
    parsed_items = json.loads(json_slots)
    print("PARSED: ", parsed_items)

    # Guardo las preguntas de los parámetros que hacen falta.
    questions = {}
    for empty in parsed_items:
        improved_question = improveQuestionchatGPT(slots[empty], domain)
        questions[empty] = improved_question

    # return questions
    return jsonify(
        {'questions': questions, 'filledslots': filledParams, 'service_id': str(service_id), 'intent': intent, 'dom': domain, 'tasks': tasks, 'final': True, 'reqslots': reqslots}), 202

def manage_task_oriented_dialogue(data):
    print("Procesando intents para dominios específicos...")
    print(data)
    questions = {}

    tasks = data.get('tasks', {})  # Obtener el diccionario de tareas
    userInput = data.get('userinput')  # Obtener el input del usuario
    service_id = data.get('service_id')  # Obtener el ID del servicio
    services = data.get('services', [])  # Obtener la lista de servicios
    userAnswers = data.get('useranswers', [])  # Obtener el historial del diálogo
    filled_slots = data.get('filledslots', {})  # Obtener los slots llenados
    reqslots = data.get('reqslots', [])  # Obtener los slots requeridos
    domain = data.get('domain')  # Obtener el dominio
    intent = data.get('intent')

    if not domain:
        # Procesar el primer dominio-intent en el diccionario de tareas
        domain = list(tasks.keys())[0]  # Obtener el primer dominio
        data['domain'] = domain  # Actualizar el dominio en el diccionario 'data'
        intent = tasks[domain]  # Obtener el intent de ese dominio
        data['intent'] = intent
        print(f"Procesando dominio: {domain}, intent: {intent}")

    # Verificar las condiciones
    if service_id:
        # Solo 'service_id' está definido, tengo un servicio seleccionado
        print("'service_id' está definido, pero 'services' no.")
        return final_slot_filling(data)
    elif services:
        # Solo 'services' está definido, necesito seleccionar el servicio entre varios posibles
        print("'services' está definido, pero 'service_id' no.")
        if len(filled_slots) > len(reqslots):
            print("Se han contestado preguntas intermedias, llamando a final_slot_filling.")
            return final_slot_filling(data)
        else:
            print("No tengo preguntas intermedias aún, llamando a service_selection.")
            return service_selection(data)
    else:
        # Ninguno de los dos está definido, es la primera interacción
        # Miro los parámetros discriminativos y genero preguntas si no los encuentro
        print("Ni 'service_id' ni 'services' están definidos. FilledSlots es: ", filled_slots)
        if filled_slots and all(value != '' for value in filled_slots.values()):
            # Si todos los slots están llenos, llamamos a service_selection
            print("Todos los slots están completos.")
            #Si tengo más slots rellenados que los que necesito, llamo a final_slot_filling ya que se han contestado preguntas intermedias
            #if len(filled_slots) > len(reqslots):
            #    print("Se han contestado preguntas intermedias, llamando a final_slot_filling.")
            #    return final_slot_filling(data)
            #else:
            #    print("No tengo preguntas intermedias aún, llamando a service_selection.")
            #    return service_selection(data)
            return service_selection(data)
        else:
            # Obtener los dos parámetros más frecuentes
            top_slots_list = get_top_parameters_combined(domain)
            print("Dos parámetros más frecuentes:", top_slots_list)

            # Inicializar el diccionario filledParams con los nombres de los slots rescatados
            filledParams = {slot['parameter']: '' for slot in top_slots_list}
            print("FilledSlots inicial:", filledParams)

            reqSlots = [slot['parameter'] for slot in top_slots_list]
            print("Slots requeridos:", reqSlots)

            # Slot filling con GPT
            slots = slotFillingGPT(userInput, reqSlots)
            print("Slots generados:", slots)

            # Verifica si slots es una cadena y convierte en lista de diccionarios
            if isinstance(slots, str):
                try:
                    slots_list = json.loads(slots)  # Convertir de cadena JSON a lista de diccionarios o diccionario
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON format for 'slots': {e}")
            else:
                slots_list = slots

            # Asegurarse de que 'slots_list' sea un diccionario o lista de diccionarios
            if isinstance(slots_list, dict):
                slots_list = [slots_list]  # Convertir el diccionario a una lista con un solo elemento
            elif not isinstance(slots_list, list):
                raise TypeError(f"Expected 'slots' to be of type 'list' or 'dict', got {type(slots_list)}")

            # Actualizamos los slots rellenados en filledParams
            try:
                for slots_dict in slots_list:
                    if isinstance(slots_dict, dict):
                        for param, value in slots_dict.items():
                            # Si el valor no es "Null", llenamos el slot en filledParams
                            if value != "Null":
                                filledParams[param] = value
            except Exception as e:
                print(f"An error occurred: {e}")

            # Generamos preguntas para los slots no rellenados (aquellos con valor "Null")
            null_params = [param for slots_dict in slots_list for param, value in slots_dict.items() if value == "Null"]

            if not null_params:
                data['filledslots'] = filledParams  # Actualizar 'filledSlots' en el diccionario 'data'
                return service_selection(data)
            else:
                # Crear una pregunta para cada slot no rellenado (valor "Null") usando GPT
                for param in null_params:
                    # Usar GPT para generar una pregunta personalizada para cada parámetro no rellenado
                    questions[param] = createQuestionGPT(param, domain)

                # Ahora usamos slots_list en lugar de slots
                update_frequencies_for_requested_slots(slots_list, reqSlots, domain)

                # Detectar y actualizar otros slots que el usuario haya mencionado pero no sean los top solicitados
                detect_and_update_other_slots(userInput, top_slots_list, domain)

                return jsonify(
                    {'questions': questions, 'filledslots': filledParams, 'intent': intent, 'userinput': userInput, 'dom': domain,
                     'reqslots': reqSlots, 'tasks': tasks, 'final': False, 'service_id':""}), 202

def send_data_to_server(data):
    tasks = data.get('tasks', {})
    if not tasks:
        print("Tasks está vacío desde el principio.")
        dialogue_history['useranswers'] = []  # Limpiar historial
        return jsonify({'end_of_conversation': True}), 202

    # Seleccionar la primera tarea
    current_domain = data['dom']
    current_intent = tasks[current_domain]
    print(f"Procesando dominio actual: {current_domain}, intent: {current_intent}")

    # Simular la llamada al servidor (puedes usar la lógica real aquí)
    #service_response = simulate_service_call(data)

    # Eliminar la tarea completada
    del tasks[current_domain]
    print(f"Tarea eliminada: {current_domain}. Tareas restantes: {tasks}")
    data['tasks'] = tasks

    # Si no hay más tareas, finalizar conversación
    if not tasks:
        print("No quedan más tareas. Finalizando conversación.")
        dialogue_history['useranswers'] = []  # Limpiar historial
        return jsonify({'end_of_conversation': True}), 202

    # Preparar datos para la siguiente tarea
    next_domain = next(iter(tasks))
    next_intent = tasks[next_domain]
    data.update({
        'domain': next_domain,
        'intent': next_intent,
        'filledslots': {},  # Reiniciar slots llenados
        'service_id': '',  # Reiniciar ID del servicio
        'useranswers': [],  # Reiniciar respuestas del usuario
        'questions': {},  # Reiniciar preguntas
        'final': False,  # Asegurarse de que no termine aún
        'reqslots': [],  # Reiniciar slots requeridos
        'userinput': data['userinput']  # Mantener el input del usuario
    })
    print("Datos preparados para la siguiente tarea:", data)

    # Llamar nuevamente a chatbot con los datos actualizados
    return chatbot(data)

def sanitize_json(slots):
    """
    Sanitiza un JSON reemplazando valores no válidos como 'Null' con 'null'.
    """
    try:
        slots_str = json.dumps(slots)
        sanitized_slots = slots_str.replace('Null', 'null')
        return json.loads(sanitized_slots)
    except Exception as e:
        raise ValueError(f"Error sanitizing slots: {e}")

@app.route('/')
def home():
    return render_template('chat-api.html')

@app.route('/chatbot', methods=['GET', 'POST'])
def chatbot(data=None):
    print("entro en domain manager con los siguientes datos: ", data)
    tasks = {}  # Diccionario para guardar intent por dominio
    global dialogue_history

    # Obtener datos del cuerpo de la solicitud POST
    data = request.get_json()  # Utiliza get_json() para obtener datos JSON
    print(data)

    # Verificar si se recibieron datos
    if not data:
        return jsonify({"error": "No data received"}), 400

    # Extraer los datos recibidos
    userInput = data.get('userinput')
    print("El userinput es: ", userInput)

    # Inicializa dialogue_history como un diccionario si no lo es
    if 'useranswers' not in dialogue_history:
        dialogue_history['useranswers'] = []  # Asegurarse de que sea una lista

    # Obtener las respuestas del usuario de los datos recibidos
    userAnswers = data.get('useranswers', [])

    # Actualizar el historial de la conversación (solo las respuestas del usuario)
    if userAnswers:
        dialogue_history['useranswers'] = userAnswers

    print("Historial del dialogo: ", dialogue_history)

    #Son preguntas finales, mando la respuesta al servidor
    if data.get('final') in [True, 'true', 'True']:
        print("Entrando en la condición 'final=True' (manejo de cadena)")
        return send_data_to_server(data)

    print("No se ha detectado el valor 'final' como True")

    # Si `tasks` ya tiene contenido, manejar la próxima tarea
    if 'tasks' in data and data['tasks']:
        print("Tasks ya está lleno, omitiendo la detección de intents.")
        return manage_task_oriented_dialogue(data)

    detected_domain = data.get('domain')
    print("domain", detected_domain)

    # Inicializar detected_domains como una lista vacía
    detected_domains = data.get('detected_domains', [])

    # Si detected_domain es "out-of-domain" o si detected_domains está vacío
    if detected_domain == "out-of-domain" or not detected_domains:
        # Reconocer dominios
        detected_domains = domain_manager_gpt(userInput)
        print("Detected domains:", detected_domains)

        # Asegurarse de que `detected_domains` es una lista
        if isinstance(detected_domains, str):
            detected_domains = [detected_domains]  # Convertir a lista si es una cadena

        # Detectar si el input pertenece a dominio abierto con patrones conocidos
        if detect_open_domain(userInput):
            print("Detectado como dominio abierto por patrón")
            detected_domains = ['out-of-domain']

    # Si se detecta un dominio abierto, redirigir a un nuevo endpoint de diálogo abierto
    if data.get('domain') == "out-of-domain" or "out-of-domain" in detected_domains:
        return manage_open_dialogue(data)
    else:
        print("se han detectado otros dominios")

        # Si `tasks` ya tiene contenido, omitir la detección de intents y continuar con el flujo
        if 'tasks' in data and data['tasks']:
            print("Tasks ya está lleno, omitiendo la detección de intents. Valor de task: ", data['tasks'])
            return manage_task_oriented_dialogue(data)

        # Inicializar tasks si está vacío o no existe
        if 'tasks' not in data or not data['tasks']:
            data['tasks'] = {}

        # Para cada dominio, reconocer el intent y almacenarlo en el diccionario tasks
        for domain in detected_domains:
            print("procesando dominio: ", domain)
            user_intent = intentRecWithChatGPT(userInput, domain)
            data['tasks'][domain] = user_intent.lower()  # Almacenar los intents en tasks

        return manage_task_oriented_dialogue(data)

if __name__ == '__main__':
    """
    from flask_cors import CORS
    import ssl

    context = ssl.SSLContext()
    context.load_cert_chain("/home/mariajesus/certificados/conversational_ugr_es.pem",
                            "/home/mariajesus/certificados/conversational_ugr_es.key")
    CORS(app)
    app.run(host='0.0.0.0', port=5050, ssl_context=context, debug=False)
    """

    app.run()