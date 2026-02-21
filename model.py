import torch  
from datasets import load_dataset
from transformers import (AutoTokenizer,AutoModelForSequenceClassification,Trainer ,TrainingArguments)
from peft import LoraConfig , get_peft_model



dataset = load_dataset('imdb')
# print(dataset['train'][0])

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def tokenize(example):
    return tokenizer(example['text'] ,
                     padding ="max_length",
                     truncation =True,
                     max_length =128
                     )
    
tokenized_dataset = dataset.map(tokenize , batched= True)
tokenized_dataset.set_format("torch")

model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased' , num_labels=2)

lora_config =LoraConfig(r =8,
                        lora_alpha=16,
                        target_modules=["query" ,"values"],
                        bias="none",
                        lora_dropout=0.1,
)

model = get_peft_model(model , lora_config)

model.print_trainable_parameters()

training_args = TrainingArguments(output_dir='sentiment_analysis/model_training_outputs/model_full_dataset',
                                  per_device_train_batch_size=64,
                                  per_device_eval_batch_size=64,
                                  num_train_epochs=10,
                                  eval_strategy="epoch",
                                  save_strategy='epoch',
                                  load_best_model_at_end=True,
)

trainer =Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'].shuffle(seed=42),
    eval_dataset=tokenized_dataset['test'].shuffle(seed=42)
)

trainer.train()

model.save_pretrained('sentiment_analysis/trained_model/model_full_dataset')
tokenizer.save_pretrained('sentiment_analysis/trained_tokenizer/model_02_full_dataset')