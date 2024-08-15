import streamlit as st
from pytube import YouTube
import cv2
from PIL import Image
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
groq_model = ChatGroq(model="llama-3.1-70b-versatile", api_key=GROQ_API_KEY)
st.title("Movie Trailer Recommendation System")
video_url = st.text_input("Enter YouTube video link:")
user_preferences = st.text_area("Enter your movie preferences:")
frames = []
descriptions = []
recommendation = None  

def download_video(url):
    yt = YouTube(url)
    stream = yt.streams.filter(progressive=True, file_extension='mp4').first()
    video_path = stream.download()
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
        # Call MetaCLIP or similar model here (placeholder)
        description = groq_model(pil_img)  # This is a placeholder
        descriptions.append(description)
    return descriptions

def recommend_movie(descriptions, user_preferences):
    input_text = f"User Preferences: {user_preferences}\nTrailer Analysis: {descriptions}"
    response = groq_model(input_text)
    return response

def fetch_cast_and_crew(video_url):
    # Placeholder for actual API that fetches cast/crew info
    # can use TMDB API (The Movie Database) for fetching such data
    return {"cast": ["Actor A", "Actor B"], "crew": ["Director X", "Producer Y"]}

if st.button("Process Trailer") and video_url:
    st.write("Downloading and processing the video...")
    video_path = download_video(video_url)
    frames = extract_frames(video_path)
    st.write(f"Extracted {len(frames)} frames from the video.")

if frames:
    descriptions = analyze_frames(frames)
    st.write("Frames analyzed successfully.")

if descriptions:
    recommendation = recommend_movie(descriptions, user_preferences)
    st.write(f"Recommendation Score: {recommendation}")

cast_crew = fetch_cast_and_crew(video_url)
st.write("Cast & Crew Information:")
st.write(cast_crew)

if recommendation:
    st.write("Final Recommendation:")
    st.write(recommendation)
    st.write("Cast & Crew Information:")
    st.write(cast_crew)
