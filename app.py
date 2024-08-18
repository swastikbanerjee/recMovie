import streamlit as st
import cv2
from PIL import Image
import os
from dotenv import load_dotenv
import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification, AutoTokenizer, AutoModelForCausalLM
from langchain_groq import ChatGroq
import requests
import yt_dlp
import warnings
import plotly.graph_objects as go
import re

warnings.filterwarnings("ignore")
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY') 
TMDB_API_KEY = os.getenv('TMDB_API_KEY')

groq_model = ChatGroq(model="llama-3.1-70b-versatile", api_key=GROQ_API_KEY) # Initialize the Groq model
image_model_name = "google/vit-base-patch16-224" # Load the image classification model
feature_extractor = AutoFeatureExtractor.from_pretrained(image_model_name)
image_model = AutoModelForImageClassification.from_pretrained(image_model_name)
text_model_name = "gpt2" # Load the text generation model
tokenizer = AutoTokenizer.from_pretrained(text_model_name)
text_model = AutoModelForCausalLM.from_pretrained(text_model_name)

st.title("CritiqueCue: A Movie Trailer Analysis App") # Streamlit UI
video_url = st.text_input("Enter YouTube video link:") # User inputs
user_preferences = st.text_area("What were your favourite movies? List them in comma separated manner: ")

def download_video(url):
    ydl_opts = {'format': 'best[ext=mp4]', 'outtmpl': '%(title)s.%(ext)s'}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_path = ydl.prepare_filename(info)
    return video_path

def extract_frames(video_path, interval=5):
    frames = []
    vidcap = cv2.VideoCapture(video_path)
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    frame_count = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    for sec in range(0, int(duration), interval):
        vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        success, image = vidcap.read()
        if success:
            frames.append(image)
    vidcap.release()
    return frames

def analyze_frames(frames):
    descriptions = []
    for frame in frames:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        description = generate_caption(pil_img)
        descriptions.append(description)
    return descriptions

@torch.no_grad()
def generate_caption(image):
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = image_model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()  
    predicted_class_label = image_model.config.id2label[predicted_class_idx] 
    return f"This image shows {predicted_class_label}"

def process_trailer(video_url, user_preferences):
    video_path = download_video(video_url)
    if video_path is None:
        st.error("Failed to download video")
        return None, None    
    frames = extract_frames(video_path)
    if not frames:
        st.error("Failed to extract frames from video")
        return None, None
    descriptions = analyze_frames(frames) 
    if not descriptions:
        st.error("Failed to analyze frames")
        return None, None    
    summary = " ".join(descriptions[:12]) # Using first 12 descriptions as a simple summary 
    # Generate analysis using Llama 3.1 70B
    analysis_text = f"""Based on the trailer summary and user preferences, generate: 
    1. A brief description of the movie trailer based on the following summary: {summary}
    2. A percent score indicating how well the film matches the user's taste.
    3. A brief visual analysis of how the film's style and content align with the user's preferences.
    4. A list of 5 films that are very similar to the summary of the trailer.
    User preferences: {user_preferences}"""
    analysis = groq_model.predict(analysis_text)
    os.remove(video_path) # Clean up downloaded video
    return descriptions, analysis

def fetch_cast_and_crew(video_url):
    with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
        info = ydl.extract_info(video_url, download=False)
        movie_title = info['title']
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_title}"
    response = requests.get(search_url).json()  
    if response['results']:
        movie_id = response['results'][0]['id']
        credits_url = f"https://api.themoviedb.org/3/movie/{movie_id}/credits?api_key={TMDB_API_KEY}"
        credits_response = requests.get(credits_url).json()    
        cast = [member['name'] for member in credits_response.get('cast', [])][:5]
        crew = [member['name'] for member in credits_response.get('crew', [])][:5]  
        return {"cast": cast, "crew": crew}
    else:
        return {"cast": [], "crew": []}

def create_gauge_chart(score):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Match Score"},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 33], 'color': "lightgray"},
                {'range': [33, 66], 'color': "gray"},
                {'range': [66, 100], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
    return fig

def extract_score(analysis):   
    match = re.search(r'(\d+)(?:\s*\/|\s*out\s*of|\s*/\s*)\s*100', analysis, re.IGNORECASE) # Try to find a score presented as "X out of 100" or similar
    if match:
        return int(match.group(1)) 
    match = re.search(r'(?:score|rating|match).*?(\d{1,3})(?:\s*\/\s*100|\s*percent|\s*%)?', analysis, re.IGNORECASE) # If not found, look for any number between 0 and 100 preceded by words like "score" or "rating"
    if match:
        score = int(match.group(1))
        return score if 0 <= score <= 100 else 0
    match = re.search(r'(\d{1,3})\s*%', analysis) # If still not found, look for any percentage
    if match:
        return int(match.group(1)) 
    return 0  # Default to 0 if no valid score is found

if st.button("Process Trailer") and video_url: # Streamlit flow
    try:
        with st.spinner("Processing trailer and generating analysis..."):
            descriptions, analysis = process_trailer(video_url, user_preferences)
        if descriptions is None or analysis is None:
            st.error("Failed to process trailer")
        else:  
            st.success("Trailer processed successfully.")
            # st.subheader("Sample Frame Descriptions:")
            # st.write(descriptions[:5])  # Show first 5 descriptions 
            st.subheader("Analysis:")
            st.write(analysis)     
            st.write("If you have liked these movies, you can definitely go for this one!")          
            score = extract_score(analysis) # Extract score and create gauge chart    
            st.subheader("Match Score Visualization:")
            fig = create_gauge_chart(score)
            st.plotly_chart(fig)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")