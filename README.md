# API-driven Multidomain Chatbot

This repository contains the implementation of an **API-driven, multidomain, task-oriented chatbot**, designed to understand user requests expressed in natural language and interact with external services described through OpenAPI specifications.

The system supports multiple domains — in this case **restaurants, hotels, and attractions** — and is capable of dynamically selecting and invoking the appropriate service depending on the user input. Unlike traditional chatbots based on rigid dialogue trees, this architecture leverages **Large Language Models (LLMs)**, specifically **GPT-3.5-turbo**, for domain and intent detection, as well as for flexible slot filling through context-aware question generation.

It is particularly suited for **modular integration of services**, allowing new service descriptions to be added or removed without retraining or reprogramming the system. The project was developed for academic and experimental purposes in the context of service-oriented dialogue systems.

## Features

-  This chatbot uses as LLM GPT-3.5-turbo
-  Extended OpenAPI (PPTalk) for service storage and invocation
-  Modular, extensible and LLM-agnostic
-  Suitable for research, prototyping, and academic evaluation

## Requirements

- Python 3.8+
- MongoDB (local or remote)
- pip for installing dependencies

> ⚠️ The OpenAI API key is currently integrated into the codebase for testing purposes and **does not need to be manually configured at this stage**.

## Installation

```bash
git clone git@github.com:mjesusrodriguez/pptalk_multidomain_chatbot.git
cd pptalk_multidomain_chatbot
pip install -r requirements.txt
```
###  Sample MongoDB Databases

This repository includes a folder named `mongo_dumps` containing example databases for each domain:

- `restaurant_services`
- `hotel_services`
- `attraction_services`

To load them into your local MongoDB instance, use the following commands:

```bash
mongorestore --db restaurant_services mongo_dumps/restaurant_services
mongorestore --db hotel_services mongo_dumps/hotel_services
mongorestore --db attraction_services mongo_dumps/attraction_services
```

> ⚠️ Make sure you have MongoDB installed locally and the `mongorestore` command available in your system.

---
## Usage

To run the chatbot API locally:

```bash
python app.py
```

Then, open your browser and navigate to:

```bash
http://localhost:5000/
```
Once the chatbot interface is open, you can start interacting by typing natural language queries. For example:

> 🗣️ “I want to eat in a vegetarian restaurant”

The chatbot will automatically detect the relevant domain (`restaurant`), infer your intent (`bookrestaurant`), and begin asking follow-up questions to gather the required information.

## Evaluation

This system was designed to support experimentation in dialog management and service integration. It can be used as a base for academic research.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.