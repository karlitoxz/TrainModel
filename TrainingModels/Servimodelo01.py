#pip install huggingface_hub transformers evaluate dataset scikit-learn accelerate -q

from huggingface_hub import list_datasets,dataset_info
from datasets import list_datasets, load_dataset,DatasetInfo
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from huggingface_hub import login
from transformers import TrainingArguments, Trainerclear

print("#Traer informacion de el dataset karlitoxz/DataSetServiefectivo")
res = dataset_info("karlitoxz/DataSetServiefectivo")
print(res)

print("#se carga el dataset karlitoxz/DataSetServiefectivo")
dataset = load_dataset("karlitoxz/DataSetServiefectivo")

small_train_dataset = dataset["train"]
small_eval_dataset = dataset["test"]

print("# se carga el modelo lxyuan/distilbert-base-multilingual-cased-sentiments-student")
modelo = "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
print(modelo)

model_path = ("./modelo")

tokenizer = AutoTokenizer.from_pretrained(model_path)

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length= 512)

small_train_dataset = small_train_dataset.map(tokenize_function, batched=True)
small_eval_dataset = small_eval_dataset.map(tokenize_function, batched=True)

print("#Comienza modelado")
model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=3)

print("#iniciar sesion en huggingface")
loginHF = login(token="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")
print(loginHF)

import numpy as np
import evaluate

metric = evaluate.load("accuracy")

print("#funcion calcular el accuracy de las predicciones")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)
    
training_args = TrainingArguments(
    'karlitoxz/ServiModel',
    evaluation_strategy="steps",
    logging_steps=3,
    num_train_epochs = 1,
    push_to_hub=True,
)

print("#instanciamos el objeto Trainer con todo lo que hemos preparado")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)

print("#Inicio de trainer.train")
trainer.train()


print("#guardar en local:")
trainer.save_model("/modelo")

print("#subir a la nube")
trainer.push_to_hub()

print("#subir tokenizer")
tokenizer.push_to_hub("karlitoxz/ServiModel")

