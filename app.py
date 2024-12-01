import streamlit as st
import os
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import pytesseract
import pyttsx3
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAI

# Path for pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# API Key
GEMINI_API_KEY = "AIzaSyB52vtL4Xalp2qoEcAhlCE69pp_B-AwPE4"  # Replace with your valid API key
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
llm = GoogleGenerativeAI(model="gemini-1.5-pro", api_key=GEMINI_API_KEY)

# YOLO Model
yolo_model = YOLO("yolov8l.pt")
engine = pyttsx3.init()

# Set up Streamlit page
st.set_page_config(page_title="Friday - Your AI Vision Partner", layout="wide")
st.markdown(
    """
    <style>
        .stApp {
            background-image: linear-gradient(to right, #a9a9a9, #e0ffff);
            color: grey;
        }
        .title {
            font-size: 2.5em;
            font-weight: bold;
            color: #000000;
            text-shadow: 2px 2px 5px #000;
        }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown("<h1 class='title'> Friday.. -  AI  Partner</h1>", unsafe_allow_html=True)

st.sidebar.title("INPUT AREA ...👇🏻")

uploaded_file = st.sidebar.file_uploader("SELECT FILE📂...", type=["jpg", "jpeg", "png"])

# Arrange buttons in a 3x2 matrix
col1, col2, col3 = st.columns(3)
col4, col5, _ = st.columns([1, 1, 1])  # Leave the last column empty for better alignment

with col1:
    scene_button = st.button("Describe Scene 👁️")
with col2:
    ocr_button = st.button("Extract Text ✍🏻")
with col3:
    tts_button = st.button("Text-to-Speech 🎧")
with col4:
    object_button = st.button("Detect Objects ❕")
with col5:
    task_button = st.button("🤖 Personalized Assistance")

# Utility functions
def extract_text_from_image(image):
    """Extracts text from the given image using OCR."""
    return pytesseract.image_to_string(image)

def text_to_speech(text):
    """Converts the given text to speech."""
    engine.say(text)
    engine.runAndWait()

def generate_scene_description(input_prompt, image_data):
    """Generates a scene description using Google Generative AI."""
    model = genai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content([input_prompt, image_data[0]])
    return response.text

def detect_objects(image):
    """Detects objects and obstacles using YOLOv8l."""
    open_cv_image = np.array(image)
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    results = yolo_model(open_cv_image)
    detected_items = []
    for result in results[0].boxes.data.tolist():
        x1, y1, x2, y2, confidence, class_id = result
        class_name = yolo_model.names[int(class_id)]
        detected_items.append(f"{class_name} ({confidence:.2f})")
    detected_objects = ", ".join(detected_items) if detected_items else "No objects detected."
    return detected_objects, results[0].plot()

def generate_task_guidance(objects, text):
    """Generates personalized guidance based on detected objects and text."""
    prompt = (
        "You are an intelligent AI assistant that provides guidance for daily tasks. Based on the following details:\n\n"
        f"Objects Detected: {', '.join(objects) if objects else 'None'}\n"
        f"Extracted Text: {text.strip() if text.strip() else 'None'}\n\n"
        "Provide actionable insights or task-specific suggestions to help visually impaired peoples. Be concise, clear, and practical."
    )
    response = llm(prompt)
    return response

# Input prompt for scene understanding
input_prompt = """
You are a highly intelligent image captioning assistant. Based on the input provided,
describe the content of the image in a detailed and humanLY.
Include details related to living and non-living things, such as animals, objects, their actions, the setting, 
and any notable features or emotions the scene conveys. Keep the description concise yet comprehensive.
"""

# Main processing
if uploaded_file:
    image = Image.open(uploaded_file)
    st.sidebar.image(image, caption="Uploaded Image", use_container_width=True)

    # Prepare image for processing
    image_data = [{"mime_type": uploaded_file.type, "data": uploaded_file.getvalue()}]

    if scene_button:
        with st.spinner("Generating scene description..."):
            scene_description = generate_scene_description(input_prompt, image_data)
            st.subheader("Scene Description")
            st.write(scene_description)

    if ocr_button:
        with st.spinner("Extracting text..."):
            text = extract_text_from_image(image)
            st.subheader("Extracted Text")
            st.write(text)

    if tts_button:
        with st.spinner("Converting text to speech..."):
            text = extract_text_from_image(image)
            if text.strip():
                text_to_speech(text)
                st.success("Text-to-Speech Conversion Completed!")
            else:
                st.warning("No text found in the image.")

    if object_button:
        with st.spinner("Detecting objects and obstacles..."):
            detected_objects, annotated_image = detect_objects(image)
            st.subheader("Detected Objects & Obstacles")
            st.write(detected_objects)
            st.image(annotated_image, caption="Detected Objects", use_container_width=True)

    if task_button:
        with st.spinner("Generating personalized assistance..."):
            objects, _ = detect_objects(image)
            text = extract_text_from_image(image)
            task_guidance = generate_task_guidance(objects.split(", "), text)
            st.subheader("Personalized Assistance")
            st.write(task_guidance)

else:
    st.write("🖼️ Upload an image from the sidebar to get started.")
