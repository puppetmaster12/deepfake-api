import subprocess
import os
from tqdm import tqdm
import argparse

class VideoFeatureExtractor:
    def __init__(self, openface_exe, input_dir, output_dir):
        self.openface_exe = openface_exe
        self.input_dir = input_dir
        self.output_dir = output_dir
    
    # Ensure output directory is available or create
    def create_output_directory(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    # Iterating through videos
    def extract_features(self):
        for video_file in tqdm(os.listdir(self.input_dir)):
            if video_file.endswith('.mp4'):
                video_path = os.path.join(self.input_dir, video_file)
                output_file = os.path.join(self.output_dir, f"{os.path.splitext(video_file)[0]}.csv")
                
                # Feature Extraction
                command = [
                    self.openface_exe,
                    "-f", video_path,
                    "-of", output_file,
                    "-q" 
                ]
                
                try:
                    subprocess.run(command, check=True)
                    print(f"Feature extraction complete")
                except subprocess.CalledProcessError as e:
                    print(f"Error: {e.stderr}")

def main():
    parser = argparse.ArgumentParser(description="A feature extraction class using OpenFace Feature Extractor")
    parser.add_argument("--openface_exe", required=True, help="Path to the OpenFace executable")
    parser.add_argument("--input_dir", required=True, help="The path to the input files")
    parser.add_argument("--output_dir", required=True, help="The path to the output where the features will be saved")
    args = parser.parse_args()
    
    extractor = VideoFeatureExtractor(args.openface_exe, args.input_dir, args.output_dir)
    extractor.extract_features()
    
if __name__=="__main__":
    main()


    


        