from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Parâmetros de otimização
load_in_4bit = True  # Usar quantização de 4 bits
max_seq_length = 24576  # Escolher qualquer valor adequado, suporte a RoPE Scaling interno
dtype = None  # Detectar automaticamente, Float16 para Tesla T4, V100, Bfloat16 para Ampere+

# Carregar o tokenizer e o modelo
model_name = "nvidia/Llama3-ChatQA-1.5-70B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Carregar o conjunto de dados
dataset = load_dataset("imdb")

# Pré-processar os dados
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Configurar treinamento
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    optim="adamw_8bit",  # Otimização em 8 bits para maior eficiência
    fp16=True,  # Ativar fp16 para economizar memória e acelerar o treinamento
    logging_steps=10,
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
eval_results = trainer.evaluate()
print(f"Accuracy: {eval_results['eval_accuracy']}")

# Salvar o modelo fine-tuned
if True:
    model.save_pretrained("model", save_method="merged_16bit")
    tokenizer.save_pretrained("model")

token_marcela = "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"

# Fazer upload para Hugging Face Hub 
if True:
    model.push_to_hub("Marseaa/Marseaa/llama3-70b-fine-tunned", save_method="merged_16bit", use_auth_token=token_marcela)
    tokenizer.push_to_hub("Marseaa/Marseaa/llama3-70b-fine-tunned", use_auth_token=token_marcela)

print("\nFim do salvamento.")