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
import time
from rouge_score import rouge_scorer


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

# Updated summarize_text_with_rouge function (no changes here, just included for clarity)
def summarize_text_with_rouge(text, model_id, reference_summary=None):
    # Start the timer to measure summarization time
    start_time = time.time()
    
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
        
        # End the timer and calculate the summarization time
        end_time = time.time()
        summarization_time = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            generated_summary = result['choices'][0]['message']['content']
            
            # ROUGE score calculation if reference summary is provided
            if reference_summary:
                scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
                scores = scorer.score(reference_summary, generated_summary)
                rouge1 = scores["rouge1"]
                rouge2 = scores["rouge2"]
                rougeL = scores["rougeL"]
                
                # Print ROUGE scores
                st.write(f"ROUGE-1: {rouge1.fmeasure:.4f}, ROUGE-2: {rouge2.fmeasure:.4f}, ROUGE-L: {rougeL.fmeasure:.4f}")
                
            return generated_summary, summarization_time
        else:
            return f"Error {response.status_code}: {response.text}", 0
    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}", 0




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
input_method = st.selectbox("Select Input Method", ["Upload PDF", "Upload Audio", "Upload Image"])

# Model selection - Available only for PDF and manual text input
if input_method in ["Upload PDF"]:
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

# Step 1: Handle PDF Upload and Summarization
if input_method == "Upload PDF":
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file:
        # Extract text from the uploaded PDF
        st.write("Extracting text from the uploaded PDF...")
        pdf_text = extract_text_from_pdf(uploaded_file)
        st.success("Text extracted successfully!")
        content = pdf_text

        # Initialize session state variables
        st.session_state['content'] = content  # Store the extracted text in session state
        st.session_state['pdf_text'] = content  # Store a copy of the full PDF text for later use

        # Summarize the extracted text only when the button is clicked
        if st.button("Summarize Text"):
            st.write("Summarizing the text...")

            # Optional: If you have a reference summary, set it here for ROUGE scoring
            reference_summary = "This is a sample reference summary for ROUGE evaluation."

            # Measure summarization time
            generated_summary, summarization_time = summarize_text_with_rouge(pdf_text, selected_model_id, reference_summary=reference_summary)

            # Store the generated summary in session state
            st.session_state['generated_summary'] = generated_summary

            # Display the summary and summarization time
            st.write("Summary:")
            st.write(generated_summary)
            st.write(f"Summarization Time: {summarization_time:.2f} seconds")

            # Convert summary to audio in English (not translated)
            tts = gTTS(text=generated_summary, lang='en')  # Use English summary for audio
            tts.save("response.mp3")
            st.audio("response.mp3", format="audio/mp3")

            st.markdown("<hr>", unsafe_allow_html=True)  # Adds a horizontal line

            # Translate the summary to the selected language
            translated_summary = translate_text(generated_summary, selected_language, selected_model_id)
            st.write(f"Translated Summary in {selected_language}:")
            st.write(translated_summary)

# Step 3: Handle Image Upload
elif input_method == "Upload Image":
    uploaded_image = st.file_uploader("Upload an image file", type=["jpg", "png"])
    
    if uploaded_image:
        st.write("Image uploaded. Extracting text using BLIP-2...")
        try:
            # Extract text using BLIP-2
            image_text = extract_text_from_image(uploaded_image)
            st.success("Text extracted successfully!")

            # Add the title "The image describes:" before the extracted text
            st.markdown("### The image describes:")
            st.markdown(f"<div style='font-size: 14px;'>{image_text}</div>", unsafe_allow_html=True)

            content = image_text  # Set the extracted text as content for further processing
        except Exception as e:
            st.error(f"Error extracting text from image: {e}")

        # Model selection for translation and Q&A
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



# Step 5: Allow real-time conversation with the chatbot
if content and selected_model_id:
    # Display the conversation history
    if "history" not in st.session_state:
        st.session_state.history = []

    for interaction in st.session_state.history:
        st.chat_message("user").write(interaction["question"])
        st.chat_message("assistant").write(interaction["response"] or "Thinking...")

    # Get user input using the chat-style input field
    user_input = st.chat_input("Ask a question about the content:")

    if user_input:
        # Set the timezone to Malaysia for the timestamp
        malaysia_tz = pytz.timezone("Asia/Kuala_Lumpur")
        current_time = datetime.now(malaysia_tz).strftime("%Y-%m-%d %H:%M:%S")

        # Prepare the interaction data for history tracking
        interaction = {
            "time": current_time,
            "input_method": "chat_input",
            "question": user_input,
            "response": "",
            "content_preview": content[:100] if content else "No content available"
        }

        # Add the user question to the history
        st.session_state.history.append(interaction)

        # Display the user's input immediately
        st.chat_message("user").write(user_input)

        # Display "Thinking..." for assistant response
        st.chat_message("assistant").write("Thinking...")

        # Track start time for response calculation
        start_time = time.time()

        # Send the question along with the content to the selected model API for the response
        url = f"{base_url}/chat/completions"
        data = {
            "model": selected_model_id,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Use the following content to answer the user's questions."},
                {"role": "system", "content": content},
                {"role": "user", "content": user_input}
            ],
            "temperature": 0.7,
            "max_tokens": 200,
            "top_p": 0.9
        }

        try:
            # Send the request to the API
            response = requests.post(url, headers=headers, json=data)

            # Track end time for response calculation
            end_time = time.time()
            response_time = end_time - start_time

            if response.status_code == 200:
                result = response.json()
                answer = result['choices'][0]['message']['content']

                # Update the latest interaction with the model's response
                st.session_state.history[-1]["response"] = answer

                # Display the assistant's response
                st.chat_message("assistant").write(answer)

                # Display the response time
                st.write(f"Response Time: {response_time:.2f} seconds")

                # Calculate ROUGE scores between the answer and the content (or summary)
                if 'generated_summary' in st.session_state:
                    reference_summary = st.session_state['generated_summary']  # The content summary

                    # Calculate ROUGE scores
                    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
                    scores = scorer.score(reference_summary, answer)
                    rouge1 = scores["rouge1"]
                    rouge2 = scores["rouge2"]
                    rougeL = scores["rougeL"]

                    # Display ROUGE scores
                    st.write(f"ROUGE-1: {rouge1.fmeasure:.4f}, ROUGE-2: {rouge2.fmeasure:.4f}, ROUGE-L: {rougeL.fmeasure:.4f}")
            else:
                st.chat_message("assistant").write(f"Error {response.status_code}: {response.text}")
        except requests.exceptions.RequestException as e:
            st.chat_message("assistant").write(f"An error occurred: {e}")


# Display the interaction history in the sidebar with clickable expanders
st.sidebar.header("Interaction History")

# Add the "Start New Chat" button to reset all session state variables
if st.sidebar.button("Start New Chat"):
    # Clear all relevant session state variables
    st.session_state['history'] = []
    st.session_state['content'] = ''
    st.session_state['generated_summary'] = ''
    st.sidebar.success("New chat started!")
    st.rerun()  # Refresh the app to reflect the changes

# Add the "Clear History" button to reset the interaction history
if st.sidebar.button("Clear History"):
    # Clear only the history while keeping other states intact
    st.session_state['history'] = []
    st.sidebar.success("History has been cleared!")
    st.rerun()  # Refresh the app to reflect the changes

# Check if there's history to display
if "history" in st.session_state and st.session_state.history:
    # Display the history with expanders
    for idx, interaction in enumerate(st.session_state.history):
        with st.sidebar.expander(f"Interaction {idx+1} - {interaction['time']}"):
            st.markdown(f"**Question:** {interaction['question']}")
            st.markdown(f"**Response:** {interaction['response']}")
            st.markdown(f"**Content Preview:** {interaction['content_preview']}")

            # Add a button to let the user pick this interaction to continue
            if st.button(f"Continue with Interaction {idx+1}", key=f"continue_{idx}"):
                # Load the selected interaction into the current session state for continuation
                st.session_state['content'] = interaction['response']  # Set the response as current content
                st.session_state['history'] = st.session_state['history'][:idx+1]  # Truncate history to this point
                st.rerun()  # Rerun the app to update the chat flow
else:
    # Display a message indicating no history is available
    st.sidebar.write("No interactions yet. Start a conversation to see history here.")



# Function to ask a question about the content
def ask_question(question):
    if question and selected_model_id:
        # Track start time for question response
        start_time = time.time()

        # Prepare the request payload for the question
        url = f"{base_url}/chat/completions"
        data = {
            "model": selected_model_id,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant. Use the following content to answer the user's questions."},
                {"role": "system", "content": st.session_state['content']},  # Use the current content as context
                {"role": "user", "content": question}
            ],
            "temperature": 0.7,
            "max_tokens": 200,
            "top_p": 0.9
        }

        try:
            # Send request to the API
            response = requests.post(url, headers=headers, json=data)
            
            # Track end time for question response
            end_time = time.time()
            response_time = end_time - start_time

            if response.status_code == 200:
                result = response.json()
                answer = result['choices'][0]['message']['content']

                # Track the interaction history
                malaysia_tz = pytz.timezone("Asia/Kuala_Lumpur")
                current_time = datetime.now(malaysia_tz).strftime("%Y-%m-%d %H:%M:%S")

                # Only store interactions with a valid question and response
                if answer and question:
                    interaction = {
                        "time": current_time,
                        "question": question,
                        "response": answer,
                        "content_preview": st.session_state['content'][:100] if st.session_state['content'] else "No content available",
                        "response_time": f"{response_time:.2f} seconds"  # Store the response time
                    }
                    if "history" not in st.session_state:
                        st.session_state.history = []
                    st.session_state.history.append(interaction)  # Add a new entry only when there's a valid response

                    # Display the answer along with the response time
                    st.write(f"Answer: {answer}")
                    st.write(f"Question Response Time: {response_time:.2f} seconds")

                    # Compute ROUGE scores for the Q&A after summarization
                    if 'generated_summary' in st.session_state:
                        reference_summary = st.session_state['generated_summary']
                        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
                        scores = scorer.score(reference_summary, answer)
                        rouge1 = scores["rouge1"]
                        rouge2 = scores["rouge2"]
                        rougeL = scores["rougeL"]

                        # Display ROUGE scores for the question-answering process
                        st.write(f"ROUGE-1: {rouge1.fmeasure:.4f}, ROUGE-2: {rouge2.fmeasure:.4f}, ROUGE-L: {rougeL.fmeasure:.4f}")

                    # Update content with the latest answer
                    st.session_state['content'] += f"\n{question}: {answer}"

            else:
                st.write(f"Error {response.status_code}: {response.text}")
        except requests.exceptions.RequestException as e:
            st.write(f"An error occurred: {e}")



