import streamlit as st
import torch
import tempfile
from pathlib import Path
import moviepy.editor as mp
from io import BytesIO
from PIL import Image, ImageDraw
from torchvision import models, transforms
import requests
from torchvision.transforms import functional as F
import numpy as np

# Streamlit page configuration
st.set_page_config(page_title="Tennis Game Tracking", layout="centered")

# Title of the application
st.title("Tennis Game Tracking")

# Hugging Face model URL
model_url = f"https://huggingface.co/chandu3094/Streamlit/resolve/main/best.torchscript"

@st.cache(allow_output_mutation=True)
def load_model():
    model = torch.hub.load_state_dict_from_url(model_url, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    return model

model = load_model()

# Initialize flags and file paths
if 'output_path' not in st.session_state:
    st.session_state.output_path = None
if 'input_file_name' not in st.session_state:
    st.session_state.input_file_name = None
if 'show_input_video' not in st.session_state:
    st.session_state.show_input_video = False
if 'show_output_video' not in st.session_state:
    st.session_state.show_output_video = False

# Layout setup: video display area on the left, buttons on the right
col1, col2 = st.columns([10, 7])

# File uploader and buttons on the right side
with col2:
    # File uploader for selecting input file
    input_file = st.file_uploader("Select Input File", type=["mp4", "mov", "avi"])

    # Set input file name if a file is uploaded
    if input_file:
        st.session_state.input_file_name = Path(input_file.name).stem  # Extract the file name without extension

    # Preview button
    if st.button("Preview Video"):
        if input_file:
            st.session_state.show_input_video = True
        else:
            st.warning("Please select a video file to preview.")

    # Process Video button
    if st.button("Process Video"):
        if input_file:
            # Create a temporary file path to store the video content
            temp_input_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

            # Write the BytesIO content to the temporary file
            with open(temp_input_path, "wb") as temp_file:
                temp_file.write(input_file.read())

            # Using MoviePy to read the video from the temporary file
            video = mp.VideoFileClip(temp_input_path)

            # Initialize a list to store processed frames
            processed_frames = []

            # Transformation: Resize to 640x640 and convert to tensor
            transform = transforms.Compose([
                transforms.Resize((640, 640)),  # Resize frames to 640x640
                transforms.ToTensor()          # Convert to PyTorch tensor
            ])

            # Processing each frame
            progress_bar = st.progress(0)
            progress_label = st.empty()  # To display progress percentage
            for i, frame in enumerate(video.iter_frames(fps=video.fps, dtype="uint8")):
                # Convert frame to PIL image
                pil_image = Image.fromarray(frame)

                # Resize and convert image to tensor
                image_tensor = transform(pil_image).unsqueeze(0)
                
                # Perform inference on the frame using YOLOv5
                with torch.no_grad():
                    results = model(image_tensor)
                
                # If the results are not in the expected format, print them to check
                print(type(results))  # This will help us understand what kind of object is returned
                
                # Access the detections and render the image
                # Assuming results is a tuple (check this in the print output)
                if isinstance(results, tuple):
                    detections = results[0]  # Access the first element of the tuple
                else:
                    detections = results  # If it's not a tuple, treat it directly as results
                
                # Check if detections is a Results object (which has render method)
                if hasattr(detections, 'render'):
                    processed_frame = detections.render()[0]  # Render and get the processed frame
                else:
                    processed_frame = frame  # If not, just use the original frame (for debugging)
                
                # Append the processed frame to the list
                processed_frames.append(processed_frame)

                # Update progress bar
                progress_percentage = int(((i + 1) / video.reader.nframes) * 100)
                progress_bar.progress(progress_percentage)
                progress_label.text(f"Processing... {progress_percentage}% complete")

            # Create a new VideoClip from processed frames
            processed_clip = mp.ImageSequenceClip(processed_frames, fps=video.fps)

            # Write the processed video to output path
            processed_clip.write_videofile(temp_input_path, codec="libx264")

            # Set output path for download
            st.session_state.output_path = temp_input_path
            st.success("Video processing complete!")
            st.session_state.show_output_video = True  # Set to True to display output video

        else:
            st.warning("Please select a video file to process.")

    # Show Output button
    if st.button("Show Output"):
        if st.session_state.output_path:
            st.session_state.show_output_video = True
        else:
            st.warning("Please process the video before showing the output.")

    # Download Output button
    st.write("Download Output:")
    if st.session_state.output_path and Path(st.session_state.output_path).exists():
        with open(st.session_state.output_path, "rb") as f:
            st.download_button(
                label="Download Processed Video",
                data=f,
                file_name=f"{st.session_state.input_file_name}_output.mp4",
                mime="video/mp4"
            )
    else:
        st.warning("No processed video available. Please upload and process a video first.")

# Video display area in the larger left column
with col1:
    # Show input video if it has been previewed
    if st.session_state.show_input_video:
        st.subheader("Input Video Preview:")
        st.video(input_file)

    # Show processed output video below the input video preview
    if st.session_state.show_output_video and st.session_state.output_path:
        st.subheader("Processed Output Video:")
        st.video(st.session_state.output_path)  # Display the processed video
