import os
import shutil

# Define source and destination paths
song_folder = 'Audio_Song_Actors_01-24'
speech_folder = 'Audio_Speech_Actors_01-24'
features_folder = 'features'

# Create the destination directory if it doesn't exist
os.makedirs(features_folder, exist_ok=True)

# Loop through actors
actors = [f"Actor_{i:02d}" for i in range(1, 25)]

for actor in actors:
    merged_actor_path = os.path.join(features_folder, actor)
    os.makedirs(merged_actor_path, exist_ok=True)

    # Paths from song and speech folders
    song_actor_path = os.path.join(song_folder, actor)
    speech_actor_path = os.path.join(speech_folder, actor)

    # Copy files from song
    if os.path.exists(song_actor_path):
        for file in os.listdir(song_actor_path):
            src = os.path.join(song_actor_path, file)
            dst = os.path.join(merged_actor_path, file)
            shutil.copy2(src, dst)

    # Copy files from speech
    if os.path.exists(speech_actor_path):
        for file in os.listdir(speech_actor_path):
            src = os.path.join(speech_actor_path, file)
            dst = os.path.join(merged_actor_path, file)
            shutil.copy2(src, dst)

print("Merging complete. Check the 'features' folder.")
