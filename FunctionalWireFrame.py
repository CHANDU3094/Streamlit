import streamlit as st
import cv2
import tempfile
import torch
import os
from pathlib import Path

# Load YOLOv5 model
MODEL_PATH = "C:/Users/gurug/yolov5/runs/train/exp4/weights/best.pt"
YOLOV5_DIR = "C:/Users/gurug/yolov5"

# Set up the YOLOv5 model for processing
def load_model():
    model = torch.hub.load(YOLOV5_DIR, 'custom', path=MODEL_PATH, source='local')
    return model

# Function to process video with YOLOv5 model
def process_video(input_video_path, model):
    # Load the input video
    cap = cv2.VideoCapture(input_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Temporary file for output video
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process video frame by frame
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Inference on the frame
        results = model(frame)

        # Draw bounding boxes and labels on the frame
        for *box, conf, cls in results.xyxy[0]:  # `results.xyxy` contains the bounding boxes
            x1, y1, x2, y2 = map(int, box)
            label = f"{model.names[int(cls)]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Write the processed frame to the output video
        out.write(frame)

        # Update progress
        processed_frames += 1
        st.session_state['progress'] = processed_frames / total_frames

    cap.release()
    out.release()
    return output_path

# Initialize session state variables
if 'progress' not in st.session_state:
    st.session_state['progress'] = 0.0
if 'output_video_path' not in st.session_state:
    st.session_state['output_video_path'] = None

# Streamlit App Layout
st.title("Tennis Game Tracking")

# Columns for layout
col1, col2 = st.columns([3, 2])

# Video display area
with col1:
    st.write("### Video")
    video_placeholder = st.empty()

# Controls area
with col2:
    input_file = st.file_uploader("Select Input File", type=["mp4", "avi", "mov"])

    # Preview the uploaded video
    if st.button("Preview Video") and input_file:
        temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        with open(temp_video_path, 'wb') as f:
            f.write(input_file.read())
        video_placeholder.video(temp_video_path)

    # Load model for processing if button clicked
    if st.button("Process Video") and input_file:
        model = load_model()

        # Save input video to a temporary file for processing
        input_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        with open(input_video_path, 'wb') as f:
            f.write(input_file.read())

        # Process video
        with st.spinner("Processing..."):
            st.session_state['output_video_path'] = process_video(input_video_path, model)
        
        st.success("Processing complete!")

    # Dynamic progress bar
    st.progress(st.session_state['progress'])

    # Show output video if available
    if st.button("Show Output") and st.session_state['output_video_path']:
        video_placeholder.video(st.session_state['output_video_path'])

    # Download output video if available
    if st.button("Download Output") and st.session_state['output_video_path']:
        with open(st.session_state['output_video_path'], "rb") as file:
            btn = st.download_button(
                label="Download Output",
                data=file,
                file_name="output_video.mp4",
                mime="video/mp4"
            )
