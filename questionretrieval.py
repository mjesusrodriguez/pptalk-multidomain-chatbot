from bson import ObjectId
from flask import jsonify

from mongo_config import MongoDB

# Obtener la base de datos
db = MongoDB()

#Le paso el intent y el servicio a esta función que representará los ENDPOINT del servicio.
#@app.route('/intentinfo/<service_id>/intent/<intent_name>')
def questionsRetrieval(service_id, intent_name, domain):
    services = db.get_collection(domain, 'services')
    intentInfo = None  # Default value
    # Busca el servicio
    document = services.find_one({"_id": ObjectId(service_id)})
    paths = document.get('paths', [])

    # Itero el JSON y saco los intents que tiene definido el servicio
    for path, methods in paths.items():
        # Quito el primer carácter del endpoint que es un /
        intent_name_without_char = path.lstrip('/')
        # Selecciono el intent que coincide con el que se le envía en la ruta
        if intent_name_without_char == intent_name:
            # Iterate over the HTTP methods for the current path
            for method, details in methods.items():
                # Check if the method is 'get' and has 'parameters'
                if method == 'get' and 'parameters' in details:
                    questions_resp = getQuestions(details.get('parameters', []))
                    intentInfo = {
                        'name': intent_name_without_char,
                        'description': details.get('description', ''),
                        'slots': questions_resp
                    }
                    # Do something with intentInfo
                    print(intentInfo)
                # Check if the method is 'post' and has 'requestBody'
                elif method == 'post' and 'requestBody' in details:
                    content = details['requestBody'].get('content', {})
                    for content_type, content_schema in content.items():
                        if 'schema' in content_schema:
                            schema_ref = content_schema['schema'].get('$ref')
                            if schema_ref:
                                schema_name = schema_ref.split('/')[-1]
                                schema = document['components']['schemas'].get(schema_name, {})
                                properties = schema.get('properties', {})
                                questions_resp = getQuestionsFromSchema(properties)
                                intentInfo = {
                                    'name': intent_name_without_char,
                                    'description': details.get('description', ''),
                                    'slots': questions_resp
                                }
                                # Do something with intentInfo
                                print(intentInfo)
            break  # Exit the loop once the intent is found
    else:
        return jsonify({'msg': 'Intent not found'}), 404

    try:
        return jsonify({'intent': intentInfo}), 202
    except:
        return jsonify({'msg': 'Error finding intent'}), 500

#cojo las preguntas de un conjunto de parámetros que se le pasa de un intent
#@app.route('/questions/<parameters>')
def getQuestions(parameters):
    slots = {}
    for i in parameters:
        if 'schema' in i and 'x-value' in i['schema']:
            continue  # Saltar este parámetro

        slots[i['name']] = i['x-custom-question']
    try:
        return slots
    except Exception:
        return jsonify({'msg': 'Error finding questions'}), 500

def getQuestionsFromSchema(properties):
    questions = []
    for name, prop in properties.items():
        question = prop.get('x-custom-question', 'No question defined')
        questions.append({
            'name': name,
            'question': question
        })
    return questions