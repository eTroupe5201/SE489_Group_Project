import pickle
from flask import Flask, request, jsonify, render_template
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

app = Flask(__name__)

# Load the model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Function to preprocess text and make predictions

def predict_sentiment(text):
    # Convert text to list if it's not already
    if not isinstance(text, list):
        text = [text]
    
    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    # Make predictions
    outputs = model(**inputs)
    predicted_labels = torch.argmax(outputs.logits, dim=1).tolist()
    print(torch.argmax(outputs.logits, dim=1).tolist())
    return predicted_labels


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['data']
    print(data)
    prediction = predict_sentiment(data)
    print(prediction)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
