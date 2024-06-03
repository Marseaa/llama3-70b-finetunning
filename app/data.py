from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

start_time = time.time()
def echo(str):
    # print time took before this call:
    print(f"[{time.time() - start_time}] {str}")

# Parâmetros de otimização
load_in_4bit = True  
max_seq_length = 24576  
dtype = None  

# Carregar o tokenizer e o modelo
model_name = "nvidia/Llama3-ChatQA-1.5-70B"
echo("Loading tokenizer and model...")
echo(f"Model name: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
echo("Tokenizer and model loaded successfully.")

# Carregar o conjunto de dados
echo("Loading dataset...")
dataset = load_dataset("imdb")
echo(f"Dataset loaded. Size: {len(dataset)}")


# Pré-processar os dados
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Configurar treinamento
training_args = TrainingArguments(
    # bote os resultados sempre dentro do diretório /data para não perder.
    # ou vc fazer commit de uma massa de dados muito grande
    output_dir="./data/results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    optim="adamw_8bit",  # Otimização em 8 bits para maior eficiência
    fp16=True,  # Ativar fp16 para economizar memória e acelerar o treinamento
    logging_steps=10,
    debug=True
)

# Dividir os dados em treinamento e avaliação
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["test"]

# Criar o trainer e treinar o modelo
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()

# Avaliar o modelo
echo("Training completed. Evaluating model...")
eval_results = trainer.evaluate()
echo(f"Evaluation results: {eval_results}")

# Salvar o modelo fine-tuned
echo("Saving model...")
if True:
    model.save_pretrained("model", save_method="merged_16bit")
    tokenizer.save_pretrained("model")
    echo("Model and tokenizer saved successfully.")

token_marcela = "seu token aqui"

# Fazer upload para Hugging Face Hub 
echo("Uploading to Hugging Face Hub...")
if True:
    model.push_to_hub("Marseaa/Marseaa/llama3-70b-fine-tunned", save_method="merged_16bit", use_auth_token=token_marcela)
    tokenizer.push_to_hub("Marseaa/Marseaa/llama3-70b-fine-tunned", use_auth_token=token_marcela)
    echo("Model and tokenizer uploaded to Hugging Face Hub.")

echo("\nFim do salvamento.")
