import json
import random
import streamlit as st
import torch
import speech_recognition as sr
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification

#Setting up the web app page
st.set_page_config(page_title="Medico Chatbot", page_icon=":robot_face:")
st.title("Welcome to the First Aid responses App")
st.markdown("I can help you with your First Aid Questions.")

# Define Confidence Threshold
confidence_threshold = 0.5

# Define a function to predict the intent of user input
def predict_intent(model, tokenizer, intent_labels, text):
    # Tokenize the input text and add special tokens
    inputs = tokenizer.batch_encode_plus([text], padding=True, truncation=True, return_tensors='pt')

    # Make a prediction
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(dim=1).item()

    # Get the predicted intent tag and the confidence score
    predicted_tag = str(intent_labels[predicted_class_idx])
    confidence_score = torch.softmax(logits, dim=1)[0][predicted_class_idx].item()

    if confidence_score < confidence_threshold:
        predicted_tag = "unknown"
    
    return predicted_tag.lower(), confidence_score


# Define a function to get a random response based on intent tag
def get_response(intents, intent_tag):
    #intent_data = next((item for item in intents if item["tag"] == intent), None)
    #intent_tag = intent["tag"].lower()
    intent_data = next((item for item in intents if item["tag"].lower() == intent_tag), None)
 
    if intent_data:
        responses = intent_data['responses']
        return random.choice(responses)
    else:
        return "I'm sorry, I didn't understand what you meant."

def transcribe_speech():
    # Create a recognizer object
    r = sr.Recognizer()

    # Use the default microphone as the audio source
    with sr.Microphone() as source:
        # Adjust for ambient noise
        r.adjust_for_ambient_noise(source)
        
        # Listen for the user's input
        audio = r.listen(source)
    
    try:
        # Use Google Speech Recognition to transcribe the audio
        text = r.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Sorry, I didn't catch that. Could you please repeat?"
    except sr.RequestError:
        return "Sorry, there was an error processing your request. Please try again."        

#Defining the main function to load the model.
@st.cache_resource
def load_model():
    # Load intents from JSON file
    with open('intents.json', 'r') as f:
        intents = json.load(f)['intents']

    # Split dataset into training and validation sets
    train_data, val_data = train_test_split(intents, test_size=0.2, random_state=42)

    # Load BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Define training data
    training_data = []
    for intent in train_data:
        for pattern in intent['patterns']:
            training_data.append((pattern, intent['tag']))

    # Shuffle training data
    random.shuffle(training_data)

    # Tokenize and encode training data
    batch_size = 16
    train_encodings = tokenizer([data[0] for data in training_data], truncation=True, padding=True, return_tensors='pt')

    # Extract all unique intent tags from the dataset
    intent_labels = list(set([data[1] for data in training_data]))

    train_labels = torch.tensor([intent_labels.index(data[1]) for data in training_data])

    # Load or create BERT model
    try:
        model = BertForSequenceClassification.from_pretrained('bert-model')
    except:
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(intent_labels))

    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Train the model
    model.train()
    for epoch in range(2):
        total_loss = 0
        for i in range(0, len(training_data), batch_size):
            inputs = {key: val[i:i+batch_size] for key, val in train_encodings.items()}
            labels = train_labels[i:i+batch_size]
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        avg_loss = total_loss / (len(training_data) / batch_size)
        print(f"Epoch: {epoch+1} Average training loss: {avg_loss}")

    # Save the trained model
    model.save_pretrained('bert-model')

    # Set the model to evaluation mode
    model.eval()

    return model, tokenizer, intent_labels, intents

# Define the Streamlit app
def app():
    # Load the model, tokenizer, and intent labels
    model, tokenizer, intent_labels, dataset = load_model()

    #Take user input via text
    user_input = st.text_input("Enter your question")

    # Create a button to start listening for speech
    if st.button("Click here to start speaking 🎤"):
        st.write("Listening...")
        user_input = transcribe_speech()
    
    st.write("User:", user_input)
    intent_tag, confidence_score = predict_intent(model, tokenizer, intent_labels, user_input)
    #st.write("Predicted tag:", intent_tag)
    #st.write("Dataset", intents)
    #st.write("Intent Labels", intent_labels)
    response = get_response(dataset, intent_tag)
    #st.write(confidence_score)
    if user_input=="":
        st.write("Bot: Hello there! What can I do for you today?")
    else:
        st.write("Bot:", response)

if __name__ == '__main__':
    app()
