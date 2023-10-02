import torch
import torch.nn as nn
import numpy as np
import json
import os
import shutil
import pandas as pd
from tqdm import tqdm
from preprocessor import VideoPreprocessor
from feature_extraction import VideoFeatureExtractor
from torch.utils.data import DataLoader, Dataset
from flask import Flask, request, jsonify
from flask import send_from_directory
from flask_cors import CORS
from sklearn.preprocessing import MinMaxScaler
from moviepy.editor import VideoFileClip
from moviepy.editor import TextClip
os.environ["IMAGEMAGICK_BINARY"] = "C:\\Program Files\\ImageMagick-7.1.1-Q16-HDRI\\magick.exe"

app = Flask(__name__)
CORS(app)

# Directory to upload videos
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to load csv files to numpy array (same as before)
def load_feature(feature_path):
    for filename in os.listdir(feature_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(feature_path, filename)
            df = pd.read_csv(file_path, index_col=0, nrows=300)
            
            frame_features = df.to_numpy()
            scaler = MinMaxScaler()
            scaled_frame_features = scaler.fit_transform(frame_features)
            return scaled_frame_features

def clear_dir():
    features_path = "features/features/"
    trim_path = "features/trimmed/"
    feature_maps = "static/feature_maps/"
    upload_path = "uploads/"

    # Remove features
    contents = os.listdir(features_path)

    for item in contents:
        item_path = os.path.join(features_path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)
        
    # Remove feature maps
    contents = os.listdir(feature_maps)

    for item in contents:
        item_path = os.path.join(feature_maps, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)
        
    # Remove trimmed videos
    contents = os.listdir(trim_path)

    for item in contents:
        item_path = os.path.join(trim_path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

    # Remove uploaded videos
    contents = os.listdir(upload_path)

    for item in contents:
        item_path = os.path.join(upload_path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

# Main LSTM model class
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout_prob=0.5):
        super(LSTMModel, self).__init__()
        self.bn = nn.BatchNorm1d(input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(p=dropout_prob)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_size),
            nn.Sigmoid()
        )
        
    # One cycle of forward propogation
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.bn(x)
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)
        return out

# Path to latest model checkpoint
MODEL_CHECKPOINT = 'checkpoints/2023-09-29_12-08-54_checkpoint.pth'
LABELS_PATH = 'features/labels/labels.json'

# Loading the model to the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(MODEL_CHECKPOINT)
input_size = checkpoint['input_size']
hidden_size = checkpoint['hidden_size']
num_layers = checkpoint['num_layers']
output_size = checkpoint['output_size']
model = LSTMModel(input_size, hidden_size, num_layers, output_size, dropout_prob=0.5)
model.load_state_dict(checkpoint['state_dict'])
model.to(device)
model.eval()

# Load the labels
with open(LABELS_PATH, "r") as json_file:
    labels_all = json.load(json_file)
    
# Function to perform inference
def lstm_infer(video_path):
    video_data = []
    trimmed_path = 'features/trimmed/'
    
    openface_exe = 'openface2/FeatureExtraction.exe'
    feature_maps = 'static/feature_maps/'
    
    features_path = 'features/features/'

    # Preprocess the video
    video_preprocessor = VideoPreprocessor(feature_maps, features_path)
    video_preprocessor.trim_video(video_path, trimmed_path)
    
    # Feature Extraction
    feature_extractor = VideoFeatureExtractor(openface_exe, trimmed_path, feature_maps)
    feature_extractor.extract_features()
    
    # Preprocess the csv file and save only the required features
    video_preprocessor.preprocess_csv()
    video_features = load_feature(features_path)
    
    # Perform inference for each frame
    video_features = torch.tensor(video_features).to(torch.float32).to(device)
    video_features = video_features.unsqueeze(0)  # Add a batch dimension

    # Perform inference on the entire video
    with torch.no_grad():
        video_data = model(video_features)

    # Calculate the final prediction for the video
    is_deepfake = video_data.item() >= 0.6
    prob = video_data.item()
    # print(prob)
    # clear_dir()
    return is_deepfake, prob

# Flask route to handle video upload and calling inference function
@app.route('/clear', methods=['POST'])
def clear_directory():
    clear_dir()
    return jsonify({"message":"Cleared"})

@app.route('/detection', methods=['POST'])
def detect_deepfake():
    if 'video' not in request.files:
        return jsonify({"error": "File has not been found"})

    file = request.files['video']

    if file.filename == '':
        return jsonify({"error": "Please select a video file"})

    if file:
        # Save the uploaded video to the temporary directory
        host_url = request.scheme + '://' + request.host
        file_url = f"{host_url}/static/feature_maps/{file.filename}"
        map_path = f'static/feature_maps/{file.filename.split(".")[0]}.avi'
        save_path = f'static/feature_maps/{file.filename}'
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        
        # print(video_path)
        file.save(video_path)
        video_dir = app.config['UPLOAD_FOLDER']
        r_p = file.filename.split('_')[0]
        
        # Perform inference
        deepfake_result, prob = lstm_infer(video_dir)
        d_message = ''
        vid_text = ''
        
        # Convert avi to mp4
        video = VideoFileClip(map_path)
        
        # if r_p == 'fake':
        #     d_message = 'Fake'
        # else:
        #     d_message = 'Real'
        # Return the result as JSON
        d_message = 'Probability of being fake - ' + str(round(prob*100, 2)) + '%'
        if deepfake_result:
            vid_text = 'Fake'
            # vid_text = d_message + ' - Confidence: ' + str(prob*100) + '%'
        else:
            vid_text = 'Real'
            # vid_text = d_message + ' - Confidence: ' + str(prob*100) + '%'
        
        # text_clip = TextClip(d_message, fontsize=100, color='red')
        # text_clip = text_clip.set_duration(video.duration)
        # video_with_text = video.set_pos(("center", "center")).set_duration(video.duration).set_opacity(1.0)
        video.write_videofile(save_path, codec='libx264', audio_codec='aac')
        
        return jsonify({"is_deepfake": d_message, 'video_path': file_url, 'vid_text': vid_text})
    

if __name__ == '__main__':
    app.run(debug=True)