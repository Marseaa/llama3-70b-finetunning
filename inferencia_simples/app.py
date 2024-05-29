from transformers import pipeline

# Carregar o modelo fine-tuned
fine_tuned_model = AutoModelForSequenceClassification.from_pretrained("./fine-tuned-bert")
fine_tuned_tokenizer = AutoTokenizer.from_pretrained("./fine-tuned-bert")

# Criar um pipeline de classificação
classifier = pipeline("sentiment-analysis", model=fine_tuned_model, tokenizer=fine_tuned_tokenizer)

# Fazer uma previsão
result = classifier("I love this movie!")
print(result)
