import streamlit as st
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

model_name = './fine-tuned-model'
tokenizer = DistilBertTokenizer.from_pretrained(model_name)
model = DistilBertForSequenceClassification.from_pretrained(model_name)

if torch.cuda.is_available():
    model.to('cuda')

st.title("Fraud Message Detection")

input_text = st.text_area("Enter the message text", "")


if st.button("Detect"):
    if input_text:
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        
        if torch.cuda.is_available():
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        
        result = "Fraud" if predictions[0] == 1 else "Not Fraud"
        st.write(f"Prediction: {result}")
    else:
        st.write("Please enter a message text.")

