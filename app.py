from flask import Flask, request, jsonify, render_template
import numpy as np
import json
import random
import pickle
from keras.models import load_model  # Import the Keras model loading function

# Initialize Flask app
app = Flask(__name__)

# Load intents from intents.json
with open('intents.json') as file:
    intents = json.load(file)

# Load your trained model
model = load_model('model.h5')  # Adjust the filename/path as necessary

with open('texts.pkl', 'rb') as file:
    words = pickle.load(file)  # List of words used in training
with open('label.pkl', 'rb') as file:
    classes = pickle.load(file) 

# Create a mapping from class to responses
class_responses = {}
for intent in intents['intents']:
    class_responses[intent['tag']] = intent['responses']

# Function to preprocess input for testing
def preprocess_input(input_text):
    pattern_words = input_text.split()
    pattern_words = [word.lower() for word in pattern_words]
    bag = [0] * len(words)  # Create bag based on the original words list
    for w in pattern_words:
        if w in words:
            bag[words.index(w)] = 1
    return np.array(bag)

# Function to test the model
def test_model(input_text):
    input_data = preprocess_input(input_text)  # Preprocess input
    prediction = model.predict(np.array([input_data]))  # Predict class
    predicted_class = classes[np.argmax(prediction)]  # Get class with highest probability
    response = class_responses.get(predicted_class, ["Maaf, saya tidak mengerti."])  # Get response list
    return random.choice(response)  # Return a random response from the list

# Define a route for the chatbot
@app.route('/chatbot', methods=['POST'])
def chatbot():
    user_input = request.json.get('message')  # Get message from JSON request
    if not user_input:
        return jsonify({'error': 'No input provided'}), 400

    result = test_model(user_input)  # Test the model using the input
    return jsonify({'response': result})  # Return the response as JSON

@app.route('/')
def home():
    return render_template('index.html') 

# Start the Flask application
if __name__ == '__main__':
    app.run(debug=True)
