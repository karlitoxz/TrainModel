#pip install transformers
#pip install flask
#pip install flask-cors

from flask import Flask, request, jsonify
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

# Cargar el modelo y el tokenizador
model_path = "modelo"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Ruta para recibir las peticiones POST
@app.route("/analizar-sentimiento", methods=["POST"])
def analizar_sentimiento():
    # Obtener el texto de la petición
    texto = request.json.get("texto", "")
    # Realizar la clasificación de sentimiento
    resultado = classifier(texto)
    # Devolver el resultado como JSON
    return jsonify(resultado)

if __name__ == "__main__":
    # Ejecutar el servidor en el puerto 5000
    app.run(host="0.0.0.0", port=5000)