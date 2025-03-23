"""
from pymongo import MongoClient
import os
from dotenv import load_dotenv

def get_mongo_client():
    load_dotenv()
    mongo_host = os.getenv("MONGO_HOST", "localhost")
    mongo_port = int(os.getenv("MONGO_PORT", 27017))
    client = MongoClient(mongo_host, mongo_port)
    return client

def get_database(db_name="services"):
    client = get_mongo_client()
    return client[db_name]

"""
from pymongo import MongoClient

class MongoDB:
    def __init__(self, uri="mongodb://localhost:27017"):
        # Inicializa la conexión a MongoDB con la URI proporcionada
        self.client = MongoClient(uri)

    def get_collection(self, db_name, collection_name):
        # Devuelve la colección de la base de datos especificada
        db = self.client[db_name]
        collection = db[collection_name]
        return collection

    def close_connection(self):
        # Cierra la conexión con MongoDB
        self.client.close()