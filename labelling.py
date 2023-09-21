import os
from tqdm import tqdm
import json
import argparse

class VideoLabeller:
    def __init__(self, video_path, label, output_path):
        self.video_path = video_path
        self.label = label
        self.output_path = output_path
        
    def celeb_label_videos(self):
        labels_dict = {}
        for filename in os.listdir(self.video_path):
            video_label = self.label
            
            if filename.endswith(".mp4"):
                key = os.path.splitext(filename)[0]
                labels_dict[key] = video_label
                
        return labels_dict
    
    def csv_label_videos(self):
        labels_dict = {}
        for filename in os.listdir(self.video_path):
            video_label = self.label
            
            if filename.endswith(".csv"):
                key = os.path.splitext(filename)[0]
                labels_dict[key] = video_label
                
        return labels_dict
    
def main():
    parser = argparse.ArgumentParser(description="Video labelling class for videos from various sources")
    parser.add_argument("--input_dir", required=True, help="The path where the videos for labelling reside")
    parser.add_argument("--label", required=False, help="The label for the videos. This applies only for videos with no metadata")
    parser.add_argument("--output_dir", required=True, help="The path to save the labels")
    parser.add_argument("--d_name", required=False, help="The name of the dataset being labelled")
    
    args = parser.parse_args()
    
    if args.d_name == "celeb":
        abs_path = os.path.abspath(args.output_dir)
        full_path = os.path.join(abs_path, "labels.json")
        
        if os.path.isfile(full_path):
            with open(full_path, "r") as json_file:
                existing_data = json.load(json_file)
                
            video_labeller = VideoLabeller(args.input_dir, args.label, args.output_dir)
            labels_dict = video_labeller.celeb_label_videos()
            
            existing_data.update(labels_dict)
            
            with open(full_path, "w") as json_file:
                json.dump(existing_data, json_file, indent=1)
    
    if args.d_name == "csv":        
        abs_path = os.path.abspath(args.output_dir)
        full_path = os.path.join(abs_path, "labels.json")
        
        if os.path.isfile(full_path):
            with open(full_path, "r") as json_file:
                existing_data = json.load(json_file)
            
            video_labeller = VideoLabeller(args.input_dir, args.label, args.output_dir)
            labels_dict = video_labeller.csv_label_videos()
            
            existing_data.update(labels_dict)
            
            with open(full_path, "w") as json_file:
                json.dump(existing_data, json_file, indent=1)

if __name__=="__main__":
    main()
            
        
            
            