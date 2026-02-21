import torch
from transformers import AutoTokenizer , AutoModelForSequenceClassification
from peft import PeftModel

tokenizer = AutoTokenizer.from_pretrained('sentiment_analysis/trained_tokenizer/model_02_full_dataset')


base_model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')


model = PeftModel.from_pretrained(base_model ,'sentiment_analysis/trained_model/model_full_dataset')


model.eval()

text = 'people where gossping about the movie even after 2 hours , the movie created very big impact and reach worlwide '

input = tokenizer(
    text,
    return_tensors = "pt",
    truncation = True,
    padding ='max_length',
    max_length = 128
)


with torch.no_grad():
    output = model(**input)
    
logits = output.logits
prob = torch.softmax(logits ,dim=1)
prediction = torch.argmax(prob , dim=1).item()
    
if prediction == 1 :
    print("Positive Review")
else :
    print("Negative Review")