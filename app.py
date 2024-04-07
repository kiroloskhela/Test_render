from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification

app = Flask(__name__)

# Load tokenizer and model
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-classification", model="kiroloskhela/Sentiment-Bert")

# Define function for sentiment analysis
def analyze_sentiment(text):
    # Tokenize input text

    
    # Perform inference
    outputs = pipe(text)
    # Get predicted label
    predicted_label = outputs[0]['label']
    predicted_score = outputs[0]['score']
    
    return predicted_label, predicted_score

@app.route('/', methods=['GET'])
def home():
    return "Welcome to Sentiment Analysis API!"

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        data = request.get_json()
        text = data['text']

        # Perform sentiment analysis
        sentiment, score = analyze_sentiment(text)

        return jsonify({
            'text': text,
            'sentiment': sentiment,
            'score': score
        })

if __name__ == '__main__':
    app.run(debug=True)
