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

chunks = []

# Hugging Face BLIP-2 Setup
hf_token = "hf_rLRfVDnchDCuuaBFeIKTAbrptaNcsHUNM"
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", token=hf_token)
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", token=hf_token)

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
# Function to split text into chunks based on the model's token limit
def split_text_into_chunks(text, max_tokens, overlap=200):
    tokens = text.split()  # Tokenize text into words
    chunks = []
    for i in range(0, len(tokens), max_tokens - overlap):
        chunk = " ".join(tokens[i:i + max_tokens])
        chunks.append(chunk)
    return chunks


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

        # Chunk the text based on the model's context length
        model_token_limits = {
            "mixtral-8x7b-32768": 5000,
            "llama-3.1-8b-instant": 20000,
            "gemma2-9b-it": 15000,
        }
        token_limit = model_token_limits[selected_model_id]
        chunks = split_text_into_chunks(pdf_text, token_limit)

        st.write(f"PDF text split into {len(chunks)} chunks for processing.")

        # Summarize each chunk
        summaries = []
        for i, chunk in enumerate(chunks):
            st.write(f"Processing chunk {i + 1} of {len(chunks)}...")
            summary = summarize_text(chunk, selected_model_id)
            summaries.append(summary)
            st.write(f"Chunk {i + 1} Summary:")
            st.write(summary)

        # Combine all summaries into a single summary
        combined_summary = " ".join(summaries)
        st.write("Combined Summary:")
        st.write(combined_summary)

        # Translate the combined summary to the selected language
        translated_summary = translate_text(combined_summary, selected_language, selected_model_id)
        st.write(f"Translated Summary in {selected_language}:")
        st.write(translated_summary)

        # Convert the combined summary to audio
        tts = gTTS(text=combined_summary, lang='en')  # Use English summary for audio
        tts.save("response.mp3")
        st.audio("response.mp3", format="audio/mp3")


# Add a progress bar for chunk processing
progress = st.progress(0)
for i, chunk in enumerate(chunks):
    st.write(f"Processing chunk {i + 1} of {len(chunks)}...")
    summary = summarize_text(chunk, selected_model_id)
    summaries.append(summary)
    progress.progress((i + 1) / len(chunks))  # Update progress bar

# Track chunk summaries in history
if "history" not in st.session_state:
    st.session_state.history = []

interaction = {
    "time": datetime.now(pytz.timezone("Asia/Kuala_Lumpur")).strftime("%Y-%m-%d %H:%M:%S"),
    "input_method": input_method,
    "chunk_summaries": summaries,
    "combined_summary": combined_summary,
    "translated_summary": translated_summary,
}
st.session_state.history.append(interaction)

# Display extracted text with adjusted font size
if uploaded_file:
    with st.expander("View Extracted Text"):
        st.markdown(f"<div style='font-size: 14px;'>{pdf_text}</div>", unsafe_allow_html=True)

    # Assign extracted text to content for chat
    content = pdf_text
else:
    st.error("Please upload a PDF file to proceed.")


 # Summarize the extracted text only when the button is clicked
if st.button("Summarize Text"):
    st.write("Summarizing the text...")
    summary = summarize_text(pdf_text, selected_model_id)
    st.write("Summary:")
    st.write(summary)

    st.markdown("<hr>", unsafe_allow_html=True)  # Adds a horizontal line

    # Translate the summary to the selected language
    translated_summary = translate_text(summary, selected_language, selected_model_id)
    st.write(f"Translated Summary in {selected_language}:")
    st.write(translated_summary)

    # Convert summary to audio in English (not translated)
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

   
    if uploaded_audio.size > 0:
        st.write("Audio file uploaded. Processing audio...")
    else:
        st.error("Uploaded audio file is empty.")


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
    # Only clear the current chat-related variables, NOT the history
    st.session_state['content'] = ''  # Clear the current content
    st.session_state['uploaded_file'] = None  # Clear any uploaded PDF
    st.session_state['uploaded_audio'] = None  # Clear any uploaded audio
    st.session_state['manual_text'] = ''  # Clear any manually entered text
    st.session_state['uploaded_image'] = None  # Clear any uploaded image
    st.session_state['selected_model_id'] = None  # Clear model selection
    st.session_state['selected_language'] = "English"  # Optionally reset the language
    
    # Optionally, reset UI components, such as resetting dropdowns or text fields
    st.rerun()  # Refresh the app to reflect the changes

# Sidebar header for the chat history
if "history" in st.session_state and st.session_state.history:
    st.sidebar.header("Interaction History")
    
    # Update sidebar history to show chunk summaries
    for idx, interaction in enumerate(st.session_state.history):
        st.sidebar.markdown(f"**{interaction['time']}**")
        st.sidebar.markdown(f"**Input Method**: {interaction['input_method']}")
        st.sidebar.markdown(f"**Combined Summary**: {interaction['combined_summary']}")
        st.sidebar.markdown("**Chunk Summaries:**")
        for chunk_summary in interaction["chunk_summaries"]:
            st.sidebar.markdown(f"- {chunk_summary}")
        st.sidebar.markdown("---")


