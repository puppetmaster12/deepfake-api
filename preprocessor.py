import os
import pandas as pd
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import argparse
import json
import shutil

class VideoPreprocessor:
    def __init__(self, feature_path, output_dir):
        self.feature_path = feature_path
        self.output_dir = output_dir
    
    # Function for preprocessing OpenFace feature maps
    def preprocess_csv(self):
        for filename in tqdm(os.listdir(self.feature_path)):
            if filename.endswith(".csv"):
                f = os.path.join(self.feature_path, filename)
            
                df = pd.read_csv(f)
                
                # Get the action unit intensity columns from the dataframe
                # action_units = df.loc[:, ' AU01_r':' AU45_r']
                # features_df = df.loc[:, ' gaze_0_x':' pose_Rz']
                
                olddf = df.iloc[:, 5:299].join(df.iloc[:, 679:695])
                # Strip whitespace
                olddf.columns = olddf.columns.str.replace(' ', '')
                
                frame_count = 300 - len(olddf)
                
                last_row = olddf.iloc[-1]
                
                new_df = pd.concat([olddf, pd.DataFrame([last_row] * frame_count)], ignore_index=True)
                
                new_df = new_df.loc[:,~new_df.columns.str.startswith('Unnamed')]
                
                output_name = os.path.join(self.output_dir, filename)
                new_df.to_csv(output_name, index=False)

    def frame_fill(self):
        # Fill row with 0 if less than 300 frames
        for filename in tqdm(os.listdir(self.feature_path)):
            if filename.endswith(".csv"):
                f = os.path.join(self.feature_path, filename)
                df = pd.read_csv(f)
                frame_count = 300 - len(df)
                
                last_row = df.iloc[-1]
                
                new_df = pd.concat([df, pd.DataFrame([last_row] * frame_count)], ignore_index=True)
                new_df = new_df.loc[:,~df.columns.str.startswith('Unnamed')]
                
                output_name = os.path.join(self.output_dir, filename)
                new_df.to_csv(output_name, index=False)
        
            
            
    def trim_video(self, videos_path, output_path):
        for filename in tqdm(os.listdir(videos_path)):
            if filename.endswith(".mp4"):
                f = os.path.join(videos_path, filename)
                output_file = os.path.join(output_path, filename)
                
                start_time = 0
                end_time = 11
                r_frames = 300
                
                video_clip = VideoFileClip(f)
                
                actual_duration = video_clip.duration
                actual_frames = video_clip.fps
                r_duration = (r_frames / actual_frames)
                
                end_time = min(start_time + r_duration, actual_duration)
                
                ffmpeg_extract_subclip(f, start_time, end_time, targetname=output_file)
    
    def balance_class(self):
        with open('datasetv1/labels/labels.json', 'r') as json_file:
            labels_file = json.load(json_file)
            
        for filename in tqdm(os.listdir(self.feature_path)):
            if filename.endswith('.csv'):
                f_path = os.path.join(self.feature_path, filename)
                r_path = os.path.join('datasetv1/features/real', filename)
                d_path = os.path.join('datasetv1/features/fake', filename)
                
                f_name = filename.split(".")[0]
                if labels_file[f_name] == '1':
                    shutil.copyfile(f_path, d_path)
                if labels_file[f_name] == '0':
                    shutil.copyfile(f_path, r_path)
        
    def extract_dfdc(self):
        meta_path = os.path.join(self.feature_path, 'metadata.json')
        abs_path = os.path.abspath(self.feature_path)
        r_path = os.path.abspath('dfdc/real')
        
        with open(meta_path, 'r') as json_file:
            meta_file = json.load(json_file)
            
        for video_name in meta_file:
            video_path = os.path.join(abs_path, video_name)
            if meta_file[video_name]['label'] == 'REAL':
                shutil.copy(video_path, r_path)
        
                
            
                
def main():
    parser = argparse.ArgumentParser(description="Preprocessing class for both csv feature maps and trimming function for videos")
    parser.add_argument("--trim", required=False, help="Used when trimming videos")
    parser.add_argument("--fill", required=False, help="Used when filling remaining frames")
    parser.add_argument("--extract", required=False, help="Used to extract real or fake videos from a directory")
    parser.add_argument("--bal", required=False, help="Used when balancing the feature classes")
    parser.add_argument("--input_dir", required=True, help="Path to the csv or video files for preprocessing")
    parser.add_argument("--output_dir", required=True, help="Path to save the preprocessed files")
    args = parser.parse_args()
    
    preprocessor = VideoPreprocessor(args.input_dir, args.output_dir)
    
    if args.trim == "true":
        preprocessor.trim_video(args.input_dir, args.output_dir)
    if args.fill == "true":
        preprocessor.frame_fill()
    if args.bal == "true":
        preprocessor.balance_class()
    if args.extract == 'true':
        preprocessor.extract_dfdc()
    else:
        preprocessor.preprocess_csv()    
    
    
if __name__ == "__main__":
    main()