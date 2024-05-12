import librosa
import numpy as np
import pandas as pd
import os
import random
import logging

#set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def extract_audio_features(file_path):
    """
    extracts audio features from the given file path
    """
    try:
        y, sr = librosa.load(file_path, sr=None)  # loads the audio file
        if y.size == 0:
            logging.warning(f"No data in {file_path}, skipping...")  # logs warning if audio data is missing
            return None

        # computes MFCCs from the audio data
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)  # averages the MFCCs

        # computes the spectral centroid
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

        # computes the zero crossing rate
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=y))

        return [mfcc_mean, spectral_centroid, zero_crossing_rate]
    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}") 
        return None

def main(metadata_path, audio_dir):
    """
    main function to load metadata, extract features, and merge them
    """
    metadata_df = pd.read_csv(metadata_path)  # loads metadata from CSV file
    metadata_df['track_id'] = metadata_df['track_id'].astype(str).str.zfill(6)  # formats 'track_id' with leading zeros

    # prepares the list of audio files
    all_files = [os.path.join(audio_dir, file) for file in os.listdir(audio_dir) if file.endswith(".mp3")]
    selected_files = random.sample(all_files, min(1000, len(all_files)))  # selects a random subset of files

    # extracts features from selected files
    features = []
    for file in selected_files:
        track_id = os.path.basename(file).split('.')[0].zfill(6)  # extracts 'track_id' from file name
        feature_values = extract_audio_features(file)  # extracts audio features
        if feature_values is not None:
            features.append([track_id] + feature_values)  # appends features and track_id to list

    # creates DataFrame from extracted features
    features_df = pd.DataFrame(features, columns=['track_id', 'MFCC', 'Spectral Centroid', 'Zero Crossing Rate'])
    merged_df = pd.merge(metadata_df, features_df, on='track_id')  # merges features with metadata

    # outputs the results
    logging.info("Merged DataFrame preview:")
    print(merged_df.head())
    print("Number of entries in merged DataFrame:", len(merged_df))

    # saves the merged DataFrame to a CSV file
    output_path = '/home/mary/Downloads/merged_data.csv'
    merged_df.to_csv(output_path, index=False)
    logging.info(f"Data saved to {output_path}")

if __name__ == "__main__":
    main('/home/mary/Downloads/metadata.csv', '/media/mary/New Volume/sample_audio')

