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
from io import BytesIO
import openai
import pytz

# Hugging Face BLIP-2 Setup
hf_token = "hf_rLRfVDnchDCuuaBFeIKTAbrptaNcsHUNM"
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", token=hf_token)
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", token=hf_token)

# Custom CSS for a more premium look, including sticky button
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

        /* Sticky Sidebar button CSS */
        .stSidebar .css-1v0m2ju {
            position: sticky;
            top: 0;  /* Make it stick to the top of the sidebar */
            z-index: 1000;  /* Ensure it stays on top of other elements */
            background-color: #282c34;  /* Light background for the sidebar */
            padding: 10px 0;  /* Some padding */
        }
        .stSidebar button {
            position: sticky;
            top: 20px;  /* Add some space from the top */
            z-index: 1000;  /* Keep it above the scrollable content */
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

# Available models, including the two new Sambanova models
available_models = {
    "Mixtral 8x7b": "mixtral-8x7b-32768",
    "Llama-3.1-8b-instant": "llama-3.1-8b-instant",
    "gemma2-9b-it": "gemma2-9b-it",
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

# Updated function to transcribe audio using the Groq Whisper API
def transcribe_audio(file):
    whisper_api_key = st.secrets["whisper"]["WHISPER_API_KEY"]  # Access Whisper API key
    url = "https://api.groq.com/openai/v1/audio/transcriptions"  # Groq transcription endpoint

    # Check file type
    valid_types = ['flac', 'mp3', 'mp4', 'mpeg', 'mpga', 'm4a', 'ogg', 'opus', 'wav', 'webm']
    extension = file.name.split('.')[-1].lower()
    if extension not in valid_types:
        st.error(f"Invalid file type: {extension}. Supported types: {', '.join(valid_types)}")
        return None

    # Prepare file buffer with proper extension in the .name attribute
    audio_data = file.read()  # Use file.read() to handle the uploaded file correctly
    buffer = BytesIO(audio_data)
    buffer.name = f"file.{extension}"  # Assigning a valid extension based on the uploaded file

    # Prepare the request payload
    headers = {"Authorization": f"Bearer {whisper_api_key}"}
    data = {"model": "whisper-large-v3-turbo", "language": "en"}

    try:
        # Send the audio file for transcription
        response = requests.post(
            url,
            headers=headers,
            files={"file": buffer},
            data=data
        )

        # Handle response
        if response.status_code == 200:
            transcription = response.json()
            return transcription.get("text", "No transcription text found.")
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error during transcription: {str(e)}")
        return None

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
input_method = st.selectbox("Select Input Method", ["Upload PDF", "Enter Text Manually", "Upload Audio", "Upload Image"])

# Model selection - Available only for PDF and manual text input
if input_method in ["Upload PDF", "Enter Text Manually"]:
    selected_model_name = st.selectbox("Choose a model:", list(available_models.keys()), key="model_selection")
    
    # Ensure that the user selects a model (no default)
    if selected_model_name:
        selected_model_id = available_models[selected_model_name]
    else:
        st.error("Please select a model to proceed.")
        selected_model_id = None
else:
    selected_model_id = None

# Sidebar for interaction history
if "history" not in st.session_state:
    st.session_state.history = []

# Initialize content variable
content = ""

# Language selection for translation
languages = [
    "English", "Chinese", "Spanish", "French", "Italian", "Portuguese", "Romanian", 
    "German", "Dutch", "Swedish", "Danish", "Norwegian", "Russian", 
    "Polish", "Czech", "Ukrainian", "Serbian", "Japanese", 
    "Korean", "Hindi", "Bengali", "Arabic", "Hebrew", "Persian", 
    "Punjabi", "Tamil", "Telugu", "Swahili", "Amharic"
]
selected_language = st.selectbox("Choose your preferred language for output", languages)

# Step 1: Handle PDF Upload
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
    else:
        st.error("Please upload a PDF file to proceed.")

    # Summarize the extracted text only when the button is clicked
    if st.button("Summarize Text"):
        st.write("Summarizing the text...")

        # Generate summary of extracted text
        summary = summarize_text(pdf_text, selected_model_id)
        st.write("Summary:")
        st.write(summary)

        st.markdown("<hr>", unsafe_allow_html=True)  # Adds a horizontal line

        # Translate the summary to the selected language
        translated_summary = translate_text(summary, selected_language, selected_model_id)
        st.write(f"Translated Summary in {selected_language}:")
        st.write(translated_summary)

        # Convert summary to audio (TTS) in English (for the summarized content)
        tts = gTTS(text=summary, lang='en')  # Use English summary for audio
        tts.save("response.mp3")
        st.audio("response.mp3", format="audio/mp3")

# Step 2: Handle Manual Text Input
elif input_method == "Enter Text Manually":
    manual_text = st.text_area("Enter your text manually:")

    if not manual_text:
        st.error("Please enter some text to proceed.")
    else:
        content = manual_text

        if st.button("Summarize Text"):
            st.write("Summarizing the entered text...")

            # Generate summary of the entered text
            summary = summarize_text(manual_text, selected_model_id)
            st.write("Summary:")
            st.write(summary)

            # Translate the summary to the selected language
            translated_summary = translate_text(summary, selected_language, selected_model_id)
            st.write(f"Translated Summary in {selected_language}:")
            st.write(translated_summary)

            # Convert summary to audio (TTS) in English (for the summarized content)
            tts = gTTS(text=summary, lang='en')  # Use English summary for audio
            tts.save("response.mp3")
            st.audio("response.mp3", format="audio/mp3")

# Step 3: Handle Image Upload
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

        # Select a model for translation and Q&A
        selected_model_name = st.selectbox("Choose a model:", list(available_models.keys()), key="model_selection")
        selected_model_id = available_models.get(selected_model_name)

# Step 4: Handle Audio Upload
elif input_method == "Upload Audio":
    uploaded_audio = st.file_uploader("Upload an audio file", type=["mp3", "wav"])

    if uploaded_audio:
        st.write("Audio file uploaded. Processing audio...")

        # Transcribe using Groq's Whisper API
        transcript = transcribe_audio(uploaded_audio)
        if transcript:
            st.write("Transcription:")
            st.write(transcript)
            content = transcript  # Set the transcription as content
        else:
            st.error("Failed to transcribe the audio.")
    else:
        st.error("Please upload an audio file to proceed.")

    # Select a model for translation and Q&A
    selected_model_name = st.selectbox("Choose a model:", list(available_models.keys()), key="audio_model_selection")
    selected_model_id = available_models.get(selected_model_name)

# Translation of the extracted text to selected language
if content:
    translated_content = translate_text(content, selected_language, selected_model_id)

    # Convert the translated content to speech (if needed)
    tts = gTTS(text=translated_content, lang='en')  # For demonstration in English
    tts.save("translated_response.mp3")
    st.audio("translated_response.mp3", format="audio/mp3")

# Step 5: Allow user to ask questions about the content (if any)
if content and selected_model_id:
    if len(st.session_state.history) == 0 or st.session_state.history[-1]["response"]:  # If the previous response is done
        question = st.text_input("Ask a question about the content:")

        if question:
            # Set the timezone to Malaysia for the timestamp
            malaysia_tz = pytz.timezone("Asia/Kuala_Lumpur")
            current_time = datetime.now(malaysia_tz).strftime("%Y-%m-%d %H:%M:%S")

            # Prepare the interaction data for history tracking
            interaction = {
                "time": current_time,
                "input_method": input_method,
                "question": question,
                "response": "",
                "content_preview": content[:100] if content else "No content available"
            }

            # Add the user question to the history
            st.session_state.history.append(interaction)

            # Send the question along with the content to the selected model API for the response
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

                    # Store the model's answer in the interaction history
                    st.session_state.history[-1]["response"] = answer

                    # Display the model's response
                    st.write(f"Answer: {answer}")

                else:
                    st.write(f"Error {response.status_code}: {response.text}")
            except requests.exceptions.RequestException as e:
                st.write(f"An error occurred: {e}")
        
    else:
        # If there's already a response from the model, ask for follow-up questions
        st.write("You can ask more questions or clarify any points.")

# Add "Start a New Chat" button to the sidebar
if st.sidebar.button("Start a New Chat"):
    # Clear all session state variables related to the current chat
    st.session_state['content'] = ''  # Clear the current extracted text
    st.session_state['uploaded_file'] = None  # Clear any uploaded PDF
    st.session_state['uploaded_audio'] = None  # Clear any uploaded audio
    st.session_state['manual_text'] = ''  # Clear any manually entered text
    st.session_state['uploaded_image'] = None  # Clear any uploaded image
    st.session_state['selected_model_id'] = None  # Clear model selection
    st.session_state['selected_language'] = "English"  # Optionally reset the language
    st.session_state['history'] = []  # Clear the chat history

    # Also clear any content that might have been summarized or translated
    st.session_state['summary'] = ''  # Clear any previously summarized text
    st.session_state['translated_summary'] = ''  # Clear any previously translated summary
    st.session_state['translated_content'] = ''  # Clear any translated content
    st.session_state['audio_file'] = None  # Clear any audio content generated for speech
    
    # Reset input method and model selection
    st.session_state['input_method'] = None
    st.session_state['selected_model_name'] = None  # Reset model selection
    st.session_state['manual_text'] = ''
    
    # Reset extracted content (which could be from PDF, image, audio)
    st.session_state['extracted_text'] = ''  # Clear extracted text from any source

    # Optionally, reset UI components like dropdowns or text fields
    st.rerun()  # Refresh the app to reflect the changes





# Sidebar header for the chat history
if "history" in st.session_state and st.session_state.history:
    st.sidebar.header("Interaction History")
    # Display each interaction with a clickable link
    for idx, interaction in enumerate(st.session_state.history):
        # Create a clickable link for each interaction in the history
        with st.sidebar.expander(f"Interaction {idx + 1} - {interaction['time']}"):
            st.markdown(f"*Input Method*: {interaction['input_method']}")
            st.markdown(f"*Question*: {interaction['question']}")
            st.markdown(f"*Response*: {interaction['response']}")
            st.markdown(f"*Content Preview*: {interaction['content_preview']}")

            # Optionally, display more details when clicking on each entry
            if interaction['response']:
                st.markdown("---")
                st.markdown(f"*Full Content*: {interaction['content_preview']}")
                st.markdown(f"*Answer*: {interaction['response']}")
                # You can add any additional details, like translation or summary
                # Example of adding the full content if it was summarized or translated
                if 'translated_summary' in interaction and interaction['translated_summary']:
                    st.markdown(f"*Translated Summary*: {interaction['translated_summary']}")
                if 'audio_file' in interaction and interaction['audio_file']:
                    st.audio(interaction['audio_file'], format="audio/mp3")
else:
    st.sidebar.write("No interactions to display.")

# Main section to allow user to view previous interaction content (click interaction to display it)
if 'history' in st.session_state and len(st.session_state.history) > 0:
    # Select interaction from history for display
    interaction_idx = st.selectbox("Select an Interaction to Review", range(len(st.session_state.history)))
    selected_interaction = st.session_state.history[interaction_idx]
    
    # Display full details of selected interaction
    st.subheader(f"Interaction {interaction_idx + 1} Details")
    st.write(f"*Time*: {selected_interaction['time']}")
    st.write(f"*Input Method*: {selected_interaction['input_method']}")
    st.write(f"*Question*: {selected_interaction['question']}")
    st.write(f"*Response*: {selected_interaction['response']}")
    st.write(f"*Content Preview*: {selected_interaction['content_preview']}")
    
    # Optionally, you can add more sections for the full content or any other detail (e.g., translations, audio, etc.)
    if 'translated_summary' in selected_interaction and selected_interaction['translated_summary']:
        st.write(f"*Translated Summary*: {selected_interaction['translated_summary']}")
    if 'audio_file' in selected_interaction and selected_interaction['audio_file']:
        st.audio(selected_interaction['audio_file'], format="audio/mp3")
