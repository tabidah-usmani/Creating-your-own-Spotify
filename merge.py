import pandas as pd
from pymongo import MongoClient
from pandas import json_normalize


def load_audio_features():
    try:
        client = MongoClient('mongodb://localhost:27017/')
        db = client["audio_features"]
        collection = db["features"]
        mongo_data = list(collection.find({}, {'_id': 0}))
        features_df = pd.DataFrame(mongo_data)
        features_df['track_id'] = features_df['audio_file'].apply(lambda x: x.split('/')[-1].split('.')[0])
        return features_df
    except Exception as e:
        print(f"Failed to load data from MongoDB: {e}")
        return pd.DataFrame()

def main():
    try:
        # Load metadata
        metadata_df = pd.read_csv('/home/mary/Downloads/metadata.csv')

        # Load features
        features_df = load_audio_features()

        if not features_df.empty:
            # Merge data
            combined_df = pd.merge(metadata_df, features_df, on='track_id', how='left')

            # Check for merge issues
            if combined_df['track_id'].isna().any():
                print("Warning: Some audio features were not matched with metadata.")

            # Print the combined DataFrame
            print(combined_df.head())

        else:
            print("No features data to merge.")
    except FileNotFoundError as fe:
        print(f"File not found: {fe}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

