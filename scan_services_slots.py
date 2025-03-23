from pymongo import MongoClient

# Conexión a MongoDB con las tres bases de datos
client = MongoClient("mongodb://localhost:27017/")
dbs = {
    "hotels": client["hotels"],
    "restaurants": client["restaurants"],
    "attractions": client["attractions"]
}

def scan_and_store_aggregated_parameters(db):
    services_collection = db["services"]
    parameter_collection = db["slot_ranking"]

    # Iterar sobre todos los servicios en la base de datos
    for service in services_collection.find():
        for path, methods in service.get("paths", {}).items():
            for method, details in methods.items():
                for param in details.get("parameters", []):
                    param_name = param.get("name")
                    x_value = param.get("schema", {}).get("x-value")  # Buscar 'x-value' en schema

                    # Solo guardar si el parámetro tiene un campo 'x-value'
                    if param_name and x_value:
                        # Actualizar el parámetro y agregar el nuevo valor si no existe en el array
                        parameter_collection.update_one(
                            {"parameter": param_name},
                            {
                                "$inc": {"service_frequency": 1},  # Incrementar frecuencia en servicios
                                "$addToSet": {"values": x_value}   # Añadir el valor si no existe
                            },
                            upsert=True  # Crear si no existe
                        )

# Escaneo de servicios en las tres bases de datos
for domain, db in dbs.items():
    print(f"Escaneando servicios en la base de datos: {domain}")
    scan_and_store_aggregated_parameters(db)
    print(f"Finalizado escaneo en {domain}")