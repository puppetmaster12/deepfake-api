import os
import shutil

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
        if item.endswith(".mp4"):
            break
        else:
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