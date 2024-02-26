from flask import Flask, render_template, request, jsonify
from transformers import BertForSequenceClassification, BertTokenizer
import torch

app = Flask(__name__)

# Load the trained model and tokenizer
model_checkpoint_path = "/home/jousha/Desktop/checkpoint-186"
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(model_checkpoint_path)

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set model to evaluation mode
model.eval()

# Define label mapping
label_mapping = {0: 'other', 1: 'castism', 2: 'money laundering', 3: 'harassment', 4: 'racism', 5: 'religion'}

# Define a function to predict label and get probabilities for new text
def predict_with_label_and_probabilities(text):
    inputs = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).squeeze().tolist()  # Convert to list
    predicted_label = torch.argmax(logits, dim=1).item()
    return predicted_label, probabilities

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text_to_predict = request.form['text_to_predict']
        predicted_label, probabilities = predict_with_label_and_probabilities(text_to_predict)
        predicted_label_name = label_mapping[predicted_label]
        return render_template('index.html', text=text_to_predict, predicted_label=predicted_label_name)

if __name__ == '__main__':
    app.run(debug=True)
