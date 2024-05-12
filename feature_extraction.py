import librosa
import numpy as np
import pandas as pd
import os
import random
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_audio_features(file_path):
    """
    Extracts audio features from the given file path.
    """
    try:
        y, sr = librosa.load(file_path, sr=None)
        if y.size == 0:
            logging.warning(f"No data in {file_path}, skipping...")
            return None

        # Compute MFCCs from the audio data
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)  # Taking mean of MFCCs for simplicity

        # Compute the spectral centroid
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

        # Compute the zero crossing rate
        zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y=y))

        return [mfcc_mean, spectral_centroid, zero_crossing_rate]
    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}")
        return None

def main(metadata_path, audio_dir):
    """
    Main function to load metadata, extract features, and merge them.
    """
    # Load metadata
    metadata_df = pd.read_csv(metadata_path)
    metadata_df['track_id'] = metadata_df['track_id'].astype(str).str.zfill(6)

    # Prepare the list of audio files
    all_files = [os.path.join(audio_dir, file) for file in os.listdir(audio_dir) if file.endswith(".mp3")]
    selected_files = random.sample(all_files, min(1000, len(all_files)))

    # Extract features from selected files
    features = []
    for file in selected_files:
        track_id = os.path.basename(file).split('.')[0].zfill(6)
        feature_values = extract_audio_features(file)
        if feature_values is not None:
            features.append([track_id] + feature_values)

    # Create DataFrame from extracted features
    features_df = pd.DataFrame(features, columns=['track_id', 'MFCC', 'Spectral Centroid', 'Zero Crossing Rate'])
    merged_df = pd.merge(metadata_df, features_df, on='track_id')

    # Output the results
    logging.info("Merged DataFrame preview:")
    print(merged_df.head())
    print("Number of entries in merged DataFrame:", len(merged_df))

    # Save the merged DataFrame to a CSV file
    output_path = '/home/mary/Downloads/merged_data.csv'
    merged_df.to_csv(output_path, index=False)
    logging.info(f"Data saved to {output_path}")

if __name__ == "__main__":
    metadata_path = '/home/mary/Downloads/metadata.csv'
    audio_dir = '/media/mary/New Volume/sample_audio'
    main(metadata_path, audio_dir)

