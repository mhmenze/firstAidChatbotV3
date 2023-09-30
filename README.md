# FirstAidChatBot
## Please install and run inside a virtual environment to make things easier.
pip install virtualenv  
virtualenv venv  
venv/Scripts/activate  

## Install these package dependencies next
pip install streamlit  
pip install torch  
pip install transformers  
pip install SpeechRecognition  
pip install scikit-learn  
pip install pyaudio  
pip install pipwin  

## Running the chatbot
To run in a streamlit web app, please add the following line to your project terminal:  
streamlit run main.py  

## Model Performance
We recommend initially to set the "epoch_value" to around 50 initially.  
After the first time since the model is saved, the accuracy keeps improving, hence we can reduce the epochs to speed up the chatbot startup.  
epoch_value in between 5-10 would be suitable after the first time.
