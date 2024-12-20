import requests
import streamlit as st
import PyPDF2
from datetime import datetime
from gtts import gTTS  # Import gtts for text-to-speech
import os
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image
import json

# Hugging Face BLIP-2 Setup
hf_token = "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"
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", use_auth_token=hf_token)
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", use_auth_token=hf_token)

# Custom CSS for a more premium look
st.markdown("""
    <style>
        .css-1d391kg {
            background-color: #1c1f24;  /* Dark background */
            color: white;
            font-family: 'Arial', sans-serif;
        }
        .css-1v0m2ju {
            background-color: #282c34;  /* Slightly lighter background */
        }
        .css-13ya6yb {
            background-color: #61dafb;  /* Button color */
            border-radius: 5px;
            padding: 10px 20px;
            color: white;
            font-size: 16px;
            font-weight: bold;
        }
        .css-10trblm {
            font-size: 18px;
            font-weight: bold;
            color: #282c34;
        }
        .css-3t9iqy {
            color: #61dafb;
            font-size: 20px;
        }
        .botify-title {
            font-family: 'Arial', sans-serif;
            font-size: 48px;
            font-weight: bold;
            color: #61dafb;
            text-align: center;
            margin-top: 50px;
            margin-bottom: 30px;
        }
    </style>
""", unsafe_allow_html=True)

# Botify Title
st.markdown('<h1 class="botify-title">Botify</h1>', unsafe_allow_html=True)

# Set up API Key from secrets
api_key = st.secrets["groq_api"]["api_key"]

# Base URL and headers for Groq API
base_url = "https://api.groq.com/openai/v1"
headers = {
    "Authorization": f"Bearer {api_key}",  # Use api_key here, not groqapi_key
    "Content-Type": "application/json"
}

# Available models
available_models = {
    "Mixtral 8x7b": "mixtral-8x7b-32768",
    "Llama 3.1 70b Versatile": "llama-3.1-70b-versatile"
}

# Step 1: Function to Extract Text from PDF
def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    extracted_text = ""
    for page in pdf_reader.pages:
        extracted_text += page.extract_text()
    return extracted_text

# Function to Summarize the Text
def summarize_text(text, model_id):
    url = f"{base_url}/chat/completions"
    data = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Summarize the following text:"},
            {"role": "user", "content": text}
        ],
        "temperature": 0.7,
        "max_tokens": 300,
        "top_p": 0.9
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return f"Error {response.status_code}: {response.text}"
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}"

# Function to Translate Text Using the Selected Model
def translate_text(text, target_language, model_id):
    url = f"{base_url}/chat/completions"
    data = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": f"Translate the following text into {target_language}."},
            {"role": "user", "content": text}
        ],
        "temperature": 0.7,
        "max_tokens": 300,
        "top_p": 0.9
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return f"Translation error: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return f"An error occurred during translation: {e}"

# Step 2: Function to Extract Text from Image using BLIP-2
def extract_text_from_image(image_file):
    # Open image from uploaded file
    image = Image.open(image_file)

    # Preprocess the image for the BLIP-2 model
    inputs = blip_processor(images=image, return_tensors="pt")

    # Generate the caption (text) for the image
    out = blip_model.generate(**inputs)
    caption = blip_processor.decode(out[0], skip_special_tokens=True)

    return caption

# Input Method Selection
input_method = st.selectbox("Select Input Method", ["Upload PDF", "Enter Text Manually", "Upload Image"])

# Model selection - Available only for PDF and manual text input
if input_method in ["Upload PDF", "Enter Text Manually"]:
    selected_model_name = st.selectbox("Choose a model:", list(available_models.keys()))
    selected_model_id = available_models[selected_model_name]
else:
    # If no model is selected, set a default value or handle accordingly
    selected_model_id = available_models["Mixtral 8x7b"]  # Set default model ID if no model is selected

# Sidebar for interaction history
if "history" not in st.session_state:
    st.session_state.history = []

# Initialize content variable
content = ""

# Language selection for translation
languages = [
    "English", "Spanish", "French", "Italian", "Portuguese", "Romanian", 
    "German", "Dutch", "Swedish", "Danish", "Norwegian", "Russian", 
    "Polish", "Czech", "Ukrainian", "Serbian", "Chinese", "Japanese", 
    "Korean", "Hindi", "Bengali", "Arabic", "Hebrew", "Persian", 
    "Punjabi", "Tamil", "Telugu", "Swahili", "Amharic"
]
selected_language = st.selectbox("Choose your preferred language for output", languages)

# Handle different input methods
if input_method == "Upload PDF":
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file:
        # Extract text from the uploaded PDF
        st.write("Extracting text from the uploaded PDF...")
        pdf_text = extract_text_from_pdf(uploaded_file)
        st.success("Text extracted successfully!")

        # Display extracted text with adjusted font size
        with st.expander("View Extracted Text"):
            st.markdown(f"<div style='font-size: 14px;'>{pdf_text}</div>", unsafe_allow_html=True)

        # Assign extracted text to content for chat
        content = pdf_text

        # Summarize the extracted text only when the button is clicked
        if st.button("Summarize Text"):
            st.write("Summarizing the text...")
            summary = summarize_text(pdf_text, selected_model_id)
            st.write("Summary:")
            st.write(summary)

            # Translate the summary to the selected language
            translated_summary = translate_text(summary, selected_language, selected_model_id)
            st.write(f"Translated Summary in {selected_language}:")
            st.write(translated_summary)

            # Convert summary to audio in English (not translated)
            tts = gTTS(text=summary, lang='en')  # Use English summary for audio
            tts.save("response.mp3")
            st.audio("response.mp3", format="audio/mp3")

elif input_method == "Enter Text Manually":
    manual_text = st.text_area("Enter your text manually:")

    if manual_text:
        # Assign entered text to content for chat
        content = manual_text

        if st.button("Summarize Text"):
            st.write("Summarizing the entered text...")
            summary = summarize_text(manual_text, selected_model_id)
            st.write("Summary:")
            st.write(summary)

            # Translate the summary to the selected language
            translated_summary = translate_text(summary, selected_language, selected_model_id)
            st.write(f"Translated Summary in {selected_language}:")
            st.write(translated_summary)

            # Convert summary to audio in English (not translated)
            tts = gTTS(text=summary, lang='en')  # Use English summary for audio
            tts.save("response.mp3")
            st.audio("response.mp3", format="audio/mp3")

elif input_method == "Upload Image":
    uploaded_image = st.file_uploader("Upload an image file", type=["jpg", "png"])

    if uploaded_image:
        st.write("Image uploaded. Extracting text using BLIP-2...")
        try:
            # Extract text using BLIP-2
            image_text = extract_text_from_image(uploaded_image)
            st.success("Text extracted successfully!")

            # Display extracted text with adjusted font size
            with st.expander("View Extracted Text"):
                st.markdown(f"<div style='font-size: 14px;'>{image_text}</div>", unsafe_allow_html=True)

            content = image_text
        except Exception as e:
            st.error(f"Error extracting text from image: {e}")

# Step 2: User Input for Questions
if content:
    question = st.text_input("Ask a question about the content:")

    if question:
        # Create interaction dictionary with timestamp
        interaction = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input_method": input_method,
            "question": question,
            "response": "",
            "content_preview": content[:100] if content else "No content available"
        }
        # Add user question to history
        st.session_state.history.append(interaction)

        if content:
            # Send the question and content to the API for response
            url = f"{base_url}/chat/completions"
            data = {
                "model": selected_model_id,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant. Use the following content to answer the user's questions."},
                    {"role": "system", "content": content},
                    {"role": "user", "content": question}
                ],
                "temperature": 0.7,
                "max_tokens": 200,
                "top_p": 0.9
            }

            try:
                response = requests.post(url, headers=headers, json=data)
                if response.status_code == 200:
                    result = response.json()
                    answer = result['choices'][0]['message']['content']

                    # Store bot's answer in the interaction history
                    interaction["response"] = answer
                    st.session_state.history[-1] = interaction

                    # Display bot's response
                    st.write("Answer:", answer)

                else:
                    st.write(f"Error {response.status_code}: {response.text}")
            except requests.exceptions.RequestException as e:
                st.write(f"An error occurred: {e}")

# Display interaction history in the sidebar
if st.session_state.history:
    st.sidebar.header("Interaction History")
    for idx, interaction in enumerate(st.session_state.history):
        st.sidebar.markdown(f"*{interaction['time']}*")
        st.sidebar.markdown(f"*Input Method*: {interaction['input_method']}")
        st.sidebar.markdown(f"*Question*: {interaction['question']}")
        st.sidebar.markdown(f"*Response*: {interaction['response']}")
        st.sidebar.markdown(f"*Content Preview*: {interaction['content_preview']}")
        st.sidebar.markdown("---")
