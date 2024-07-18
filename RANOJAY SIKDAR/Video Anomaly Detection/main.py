import streamlit as st
import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from joblib import load
import os
from moviepy.editor import VideoFileClip, ImageSequenceClip

# Define the fixed length for feature vectors
FIXED_LENGTH = 1000

# Paths to the trained model and scaler
model_path = 'social_force_model_UCF.joblib'
scaler_path = 'scaler_UCF.joblib'

# Use st.cache_resource to load the model and scaler only once
@st.cache_resource
def load_model_and_scaler():
    """Load the pre-trained OneClassSVM model and the StandardScaler."""
    svm = load(model_path)
    scaler = load(scaler_path)
    return svm, scaler

# Load the model and scaler
svm, scaler = load_model_and_scaler()

def compute_optical_flow(video_path, progress):
    """
    Compute the dense optical flow for the input video.
    
    Parameters:
    - video_path: Path to the input video file.
    - progress: Streamlit progress bar to display progress.
    
    Returns:
    - flow_list: List of optical flow matrices for each frame.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Error: Could not open video file {video_path}")
        return None

    ret, frame1 = cap.read()
    if not ret:
        st.error(f"Error: Failed to read the first frame from {video_path}")
        cap.release()
        return None

    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    flow_list = []
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame2 = cap.read()
        if not ret:
            break
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_list.append(flow)
        prvs = next

        frame_count += 1
        progress.progress(frame_count / total_frames)

    cap.release()
    return flow_list

def extract_features(frame, prvs):
    """
    Extract features from the frame using optical flow and social forces.
    
    Parameters:
    - frame: Current video frame.
    - prvs: Previous grayscale frame.
    
    Returns:
    - features: Feature vector of fixed length.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prvs, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = flow.shape[:2]
    force_matrix = np.zeros((h, w, 2))

    force_matrix[:, :, 0] += flow[:, :, 0]
    force_matrix[:, :, 1] += flow[:, :, 1]

    features = np.sqrt(np.sum(np.square(force_matrix), axis=2)).flatten()

    # Resize or pad the feature vector to the fixed length
    if len(features) > FIXED_LENGTH:
        features = features[:FIXED_LENGTH]
    else:
        features = np.pad(features, (0, FIXED_LENGTH - len(features)), 'constant')

    return features

def logistic_function(x):
    """Compute logistic function for a given input x."""
    return 1 / (1 + np.exp(-x))

def process_video(input_video_path):
    """
    Process the video to detect anomalies.
    
    Parameters:
    - input_video_path: Path to the input video file.
    
    Returns:
    - processed_frames: List of processed video frames with anomaly labels.
    - fps: Frames per second of the input video.
    """
    if not os.path.exists(input_video_path):
        st.error(f"Error: Video file {input_video_path} does not exist.")
        return None

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        st.error(f"Error: Could not open video file {input_video_path}")
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    processed_frames = []

    frame_count = 0
    progress = st.progress(0)
    flow_list = compute_optical_flow(input_video_path, progress)
    if flow_list is None:
        st.error(f"Error: Failed to compute optical flow for {input_video_path}")
        return None

    ret, frame = cap.read()
    if ret:
        prvs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        st.error(f"Error: Could not read the first frame from {input_video_path}")
        cap.release()
        return None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Extract features from the current frame
        features = extract_features(frame, prvs)
        features = scaler.transform([features])

        # Predict using the trained model
        prediction = svm.predict(features)
        decision_score = svm.decision_function(features)[0]
        confidence_score = logistic_function(decision_score)  # Convert to range 0 to 1

        # Determine the label and color
        if confidence_score > 0.5:
            label = f"Anomaly"
            color = (0, 0, 255)  # Red
        else:
            label = f"Normal"
            color = (0, 255, 0)  # Green

        # Put the label on the frame
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        processed_frames.append(frame)

        frame_count += 1
        progress.progress(frame_count / total_frames)

    cap.release()
    progress.empty()
    return processed_frames, fps

def save_video(frames, output_path, fps):
    """
    Save the processed frames as a video.
    
    Parameters:
    - frames: List of processed frames.
    - output_path: Path to save the output video.
    - fps: Frames per second for the output video.
    """
    clip = ImageSequenceClip([cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames], fps=fps)
    clip.write_videofile(output_path, codec='libx264')

# Streamlit app layout
def main():
    """Main function to run the Streamlit app."""
    st.title("Anomaly Detection in Video")

    uploaded_file = st.file_uploader("Upload your video for anomaly detection", type=["mp4", "avi", "mov", "mkv", "flv"])

    if uploaded_file is not None:
        # Save uploaded file to disk
        temp_input_path = 'uploaded_video_temp'
        file_extension = uploaded_file.name.split('.')[-1]
        input_video_path = f'{temp_input_path}.{file_extension}'
        with open(input_video_path, "wb") as f:
            f.write(uploaded_file.read())

        # Ensure compatibility with different video formats
        try:
            with VideoFileClip(input_video_path) as video:
                video.write_videofile('temp_video.mp4')
            input_video_path = 'temp_video.mp4'
        except Exception as e:
            st.error(f"Error: Could not process video file. {e}")
            return

        # Create a green "Start Analyze" button
        analyze_button = st.markdown(
            """
            <style>
            div.stButton > button:first-child {
                background-color: #00cc00;
                color: white;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        if st.button("Start Analyze"):
            st.write("Analyzing video for anomalies...")
            processed_frames, fps = process_video(input_video_path)

            if processed_frames:
                st.write("Analysis complete. Displaying processed video and results...")

                output_video_path = 'output_video.mp4'
                save_video(processed_frames, output_video_path, fps)

                st.video(output_video_path)

                if st.button("Close"):
                    # Delete temporary files
                    if os.path.exists(output_video_path):
                        os.remove(output_video_path)
                    if os.path.exists("temp_video.mp4"):
                        os.remove("temp_video.mp4")
                    if os.path.exists(f"{temp_input_path}.{file_extension}"):
                        os.remove(f"{temp_input_path}.{file_extension}")

                    st.write("Files deleted. You can now close this tab.")

if __name__ == "__main__":
    main()