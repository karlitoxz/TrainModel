#pip install transformers
#pip install flask
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification
model_path = ("./modelo")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)


classifier = pipeline ("sentiment-analysis",model=model,tokenizer=tokenizer)
res = classifier("el profesor es bueno todo sera sencillo")
print(res)