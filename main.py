import json
import random
import streamlit as st
import torch
import speech_recognition as sr
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Setting up the web app page
st.set_page_config(page_title="Medico Chatbot", page_icon=":robot_face:")
st.title("Welcome to First Aid Recommendation Chatbot:alien:")
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


# Defining the main function to load the model.
@st.cache_resource
def load_model():
    # Load intents from JSON file
    with open('intents.json', 'r') as f:
        dataset = json.load(f)['intents']

    # Split dataset into training and test sets
    train_data, test_data = train_test_split(dataset, test_size=0.1, random_state=42)

    # Combine training and test data
    combined_data = train_data + test_data

    # Split training data into training and validation sets
    train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)

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
    epoch_size = 1
    train_encodings = tokenizer([data[0] for data in training_data], truncation=True, padding=True, return_tensors='pt')

    # Extract all unique intent tags from the combined dataset
    intent_labels = list(set([data['tag'] for data in combined_data]))

    train_labels = torch.tensor([intent_labels.index(data[1]) for data in training_data])

    # Load or create BERT model
    try:
        model = BertForSequenceClassification.from_pretrained('bert-model')
    except:
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(intent_labels))

    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Early stopping variables
    patience = 3
    best_loss = float('inf')
    epochs_without_improvement = 0

    # Train the model
    model.train()
    for epoch in range(epoch_size):
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

        # Check for early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # Save the trained model
    model.save_pretrained('bert-model')

    # Set the model to evaluation mode
    model.eval()
    #test_patterns = [data['patterns'] for data in test_data]
    test_patterns = [data['patterns'][0] for data in test_data]
    test_encodings = tokenizer(
        test_patterns,
        truncation=True,
        padding=True,
        return_tensors='pt'
    )
    test_inputs = (test_encodings['input_ids'], test_encodings['attention_mask'])
    test_labels = ([data['tag'] for data in test_data])
    # test_outputs = model(*test_inputs, labels=None)
    # test_predictions_indices = test_outputs.logits.argmax(dim=1).tolist()

    test_outputs = model(**test_encodings, labels=None)
    test_predictions_indices = test_outputs.logits.argmax(dim=1).tolist()


    # Convert predictions from indices to labels
    test_predictions = [intent_labels[idx] for idx in test_predictions_indices if idx < len(intent_labels)]

    return model, tokenizer, intent_labels, train_data, val_data, test_data, test_labels, test_predictions

# Define the Streamlit app
def app():
    # Load the model, tokenizer, and intent labels
    model, tokenizer, intent_labels, train_data, _, test_data, test_labels, test_predictions = load_model()

    # Define a Boolean variable to track whether to show the performance metrics
    show_metrics = False

    # Create a sidebar for the performance metrics
    st.sidebar.title("**Performance Metrics**")

    # Create a button to show the performance metrics
    show_metrics_button = st.sidebar.button("Show Performance Metrics")

    st.write("\n\n")
    # Take user input via text
    user_input = st.text_input(":orange[_Type Here_]")

    # Create a button to start listening for speech
    if st.button("Click here to start speaking 🎤"):
        st.write("_Listening..._")
        user_input = transcribe_speech()

    st.write("\n")
    st.write(f":orange[**User:** {user_input}]")
    intent_tag, confidence_score = predict_intent(model, tokenizer, intent_labels, user_input)
    response = get_response(train_data, intent_tag) 
    if user_input == "":
        st.write(":red[**Bot** :robot_face::] Hello there! What can I do for you today?")
    else:
        st.write(":red[**Bot** :robot_face::]", response)

    # Check if show_metrics is True and display/hide the evaluation metrics
    if show_metrics_button:
        # Calculate evaluation metrics
        accuracy = accuracy_score(test_labels, test_predictions)
        precision = precision_score(test_labels, test_predictions, average='macro')
        recall = recall_score(test_labels, test_predictions, average='macro')
        f1 = f1_score(test_labels, test_predictions, average='macro')

        st.sidebar.write("Confidence Score:", confidence_score)
        st.sidebar.write("Accuracy:", accuracy)
        st.sidebar.write("Precision:", precision)
        st.sidebar.write("Recall:", recall)
        st.sidebar.write("F1-score:", f1)

        show_metrics = not show_metrics

    with st.expander("Classification Report:", expanded=False):
        report = classification_report(test_labels, test_predictions, zero_division=0, output_dict=True)
        st.write(report)

       #######
    st.write(test_predictions)
    st.write(test_labels)
    st.write(intent_labels)

if __name__ == '__main__':
    app()