"""
import torch
from datasets import load_dataset, Dataset
from transformers import ConvBertTokenizer, ConvBertForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm import tqdm
import os  # Para crear carpetas de checkpoints

# Cargar MultiWOZ y filtrar los dominios de interés: restaurant, hotel, attraction
def filter_domains(example):
    domains_of_interest = ['restaurant', 'hotel', 'attraction']
    return any(domain in example['services'] for domain in domains_of_interest)

# Cargar el dataset MultiWOZ desde Hugging Face
multiwoz_data = load_dataset("multi_woz_v22")
filtered_train = list(filter(filter_domains, multiwoz_data['train']))

# Etiquetar los dominios: restaurant=0, hotel=1, attraction=2
def label_domains(example):
    if 'restaurant' in example['services']:
        return 0  # Restaurant
    elif 'hotel' in example['services']:
        return 1  # Hotel
    elif 'attraction' in example['services']:
        return 2  # Attraction

# Extraer los diálogos y las etiquetas
data = {
    "text": [utterance for example in filtered_train for utterance in example['turns']['utterance']],
    "labels": [label_domains(example) for example in filtered_train for _ in example['turns']['utterance']]
}

# Convertir los datos a un Dataset de Hugging Face
dataset = Dataset.from_dict(data)

# Dividir el dataset en entrenamiento y validación (80% entrenamiento, 20% validación)
train_test = dataset.train_test_split(test_size=0.2)
train_dataset = train_test["train"]
valid_dataset = train_test["test"]

# Revisar la distribución de etiquetas en el conjunto de entrenamiento
from collections import Counter
label_distribution = Counter(train_dataset['labels'])
print(f"Distribución de etiquetas en el conjunto de entrenamiento: {label_distribution}")

# Cargar el tokenizador y el modelo ConvBERT preentrenado
tokenizer = ConvBertTokenizer.from_pretrained("YituTech/conv-bert-small")
model = ConvBertForSequenceClassification.from_pretrained("YituTech/conv-bert-small", num_labels=3)

# Tokenización de los datos
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

# Tokenizar el dataset
train_dataset = train_dataset.map(tokenize_function, batched=True)
valid_dataset = valid_dataset.map(tokenize_function, batched=True)

# Elimina las columnas de texto originales ya que ya tenemos las entradas tokenizadas
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
valid_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Convertir el dataset tokenizado a DataLoader para PyTorch
train_loader = DataLoader(train_dataset, batch_size=16)
valid_loader = DataLoader(valid_dataset, batch_size=16)

# Configuración del optimizador
optimizer = AdamW(model.parameters(), lr=2e-5)

# Scheduler para ajustar la tasa de aprendizaje durante el entrenamiento
num_epochs = 3
num_training_steps = num_epochs * len(train_loader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# Entrenar el modelo
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Usar tqdm para mostrar una barra de progreso durante el entrenamiento
model.train()
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

    for step, batch in enumerate(progress_bar):
        # Mover los tensores a la GPU (si está disponible)
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        running_loss += loss.item()

        # Mostrar la pérdida en la barra de progreso
        progress_bar.set_postfix({'loss': loss.item()})

        # Guardar cada 1000 batches
        if (step + 1) % 500 == 0:
            checkpoint_dir = f"./trained_model_epoch_{epoch + 1}_step_{step + 1}"
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Guardar los pesos del modelo manualmente con torch.save
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "pytorch_model.bin"))

            # Guardar la configuración del modelo
            model.config.save_pretrained(checkpoint_dir)

            # Guardar el tokenizador
            tokenizer.save_pretrained(checkpoint_dir)

            print(f"Checkpoint guardado en '{checkpoint_dir}'.")

    avg_loss = running_loss / len(train_loader)
    print(f"  Epoch {epoch + 1} finished with average loss: {avg_loss:.4f}\n")

    # Asegurarse de que los tensores no contiguos se conviertan en contiguos
    for param in model.parameters():
        if not param.is_contiguous():
            param.data = param.contiguous()

    # Guardar el modelo y el tokenizador al final de cada época
    checkpoint_dir = f"./trained_model_epoch_{epoch+1}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    print(f"Modelo y tokenizador guardados en '{checkpoint_dir}'.")

# Evaluar el modelo en el conjunto de validación
model.eval()
total_eval_loss = 0
correct_predictions = 0
total_predictions = 0

for batch in valid_loader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits

    total_eval_loss += loss.item()
    predictions = torch.argmax(logits, dim=-1)
    correct_predictions += (predictions == batch["labels"]).sum().item()
    total_predictions += predictions.shape[0]

avg_eval_loss = total_eval_loss / len(valid_loader)
accuracy = correct_predictions / total_predictions

print(f"Validation Loss: {avg_eval_loss}")
print(f"Validation Accuracy: {accuracy}")

# Inference: Predecir el dominio de una nueva entrada del usuario
def predict_domain(input_text):
    model.eval()
    inputs = tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=-1).item()

    label_map = {0: "restaurant", 1: "hotel", 2: "attraction"}
    return label_map[predicted_label]

# Probar con un ejemplo de entrada del usuario
input_text = "I want to book a room in a hotel."
predicted_domain = predict_domain(input_text)
print(f"Input: {input_text}")
print(f"Predicted Domain: {predicted_domain}")
"""
import torch
from datasets import load_dataset, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
import torch.nn.functional as F
import os


# Cargar MultiWOZ y filtrar los dominios de interés: restaurant, hotel, attraction
def filter_domains(example):
    domains_of_interest = ['restaurant', 'hotel', 'attraction']
    return any(domain in example['services'] for domain in domains_of_interest)


# Cargar el dataset MultiWOZ desde Hugging Face
multiwoz_data = load_dataset("multi_woz_v22")
filtered_train = list(filter(filter_domains, multiwoz_data['train']))


# Etiquetar los dominios: restaurant=0, hotel=1, attraction=2
def label_domains(example):
    if 'restaurant' in example['services']:
        return 0  # Restaurant
    elif 'hotel' in example['services']:
        return 1  # Hotel
    elif 'attraction' in example['services']:
        return 2  # Attraction


# Extraer los diálogos y las etiquetas
data = {
    "text": [utterance for example in filtered_train for utterance in example['turns']['utterance']],
    "labels": [label_domains(example) for example in filtered_train for _ in example['turns']['utterance']]
}

# Convertir los datos a un Dataset de Hugging Face
dataset = Dataset.from_dict(data)

# Dividir el dataset en entrenamiento y validación (80% entrenamiento, 20% validación)
train_test = dataset.train_test_split(test_size=0.2)
train_dataset = train_test["train"]
valid_dataset = train_test["test"]

# Revisar la distribución de etiquetas en el conjunto de entrenamiento
from collections import Counter

label_distribution = Counter(train_dataset['labels'])
print(f"Distribución de etiquetas en el conjunto de entrenamiento: {label_distribution}")

# Definir el dispositivo (CPU o GPU)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Pesos ajustados para la función de pérdida en base a la distribución de clases
class_weights = torch.tensor([1.0, 1.2, 2.0]).to(device)
loss_fn = CrossEntropyLoss(weight=class_weights)

# Cargar el tokenizador y el modelo BERT preentrenado
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
model.to(device)


# Tokenización de los datos
def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)


# Tokenizar el dataset
train_dataset = train_dataset.map(tokenize_function, batched=True)
valid_dataset = valid_dataset.map(tokenize_function, batched=True)

# Elimina las columnas de texto originales ya que ya tenemos las entradas tokenizadas
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
valid_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Convertir el dataset tokenizado a DataLoader para PyTorch
train_loader = DataLoader(train_dataset, batch_size=16)
valid_loader = DataLoader(valid_dataset, batch_size=16)

# Configuración del optimizador
optimizer = AdamW(model.parameters(), lr=2e-5)

# Scheduler para ajustar la tasa de aprendizaje durante el entrenamiento
num_epochs = 5
num_training_steps = num_epochs * len(train_loader)
lr_scheduler = get_scheduler(
    name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
)

# Entrenar el modelo
model.train()
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    running_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

    for step, batch in enumerate(progress_bar):
        # Mover los tensores a la GPU (si está disponible)
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass con la función de pérdida ponderada
        outputs = model(**batch)
        loss = loss_fn(outputs.logits, batch['labels'])
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        running_loss += loss.item()

        # Mostrar la pérdida en la barra de progreso
        progress_bar.set_postfix({'loss': loss.item()})

        # Guardar cada 1000 batches
        if (step + 1) % 1000 == 0:
            checkpoint_dir = f"./trained_model_epoch_{epoch + 1}_step_{step + 1}"
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Guardar los pesos del modelo manualmente con torch.save
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "pytorch_model.bin"))

            # Guardar la configuración del modelo
            model.config.save_pretrained(checkpoint_dir)

            # Guardar el tokenizador
            tokenizer.save_pretrained(checkpoint_dir)

            print(f"Checkpoint guardado en '{checkpoint_dir}'.")

    avg_loss = running_loss / len(train_loader)
    print(f"  Epoch {epoch + 1} finished with average loss: {avg_loss:.4f}\n")

# Evaluar el modelo en el conjunto de validación
model.eval()
total_eval_loss = 0
correct_predictions = 0
total_predictions = 0

for batch in valid_loader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits

    total_eval_loss += loss.item()
    predictions = torch.argmax(logits, dim=-1)
    correct_predictions += (predictions == batch["labels"]).sum().item()
    total_predictions += predictions.shape[0]

avg_eval_loss = total_eval_loss / len(valid_loader)
accuracy = correct_predictions / total_predictions

print(f"Validation Loss: {avg_eval_loss}")
print(f"Validation Accuracy: {accuracy}")


# Inference: Predecir el dominio de una nueva entrada del usuario con umbral para out-of-domain
def predict_domain_with_threshold(input_text, threshold=0.6):
    model.eval()
    inputs = tokenizer(input_text, return_tensors="pt", padding="max_length", truncation=True, max_length=128).to(
        device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        # Convertir logits a probabilidades
        probabilities = F.softmax(logits, dim=-1)

        # Obtener la probabilidad más alta y la etiqueta correspondiente
        max_prob, predicted_label = torch.max(probabilities, dim=-1)

        # Si la probabilidad es menor que el umbral, clasificamos como "out-of-domain"
        if max_prob.item() < threshold:
            return "out-of-domain"

        # Mapear la etiqueta predicha al dominio
        label_map = {0: "restaurant", 1: "hotel", 2: "attraction"}
        return label_map[predicted_label.item()]


# Probar con un ejemplo de entrada del usuario
input_text = "I want to book a room in a hotel."
predicted_domain = predict_domain_with_threshold(input_text)
print(f"Input: {input_text}")
print(f"Predicted Domain: {predicted_domain}")