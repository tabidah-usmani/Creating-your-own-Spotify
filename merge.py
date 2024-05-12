import pandas as pd  
from pymongo import MongoClient

def load_audio_features():
    try:
        client = MongoClient('mongodb://localhost:27017/')  # connects to the local MongoDB server
        db = client["audio_features"]  # accesses the 'audio_features' database
        collection = db["features"]  # accesses the 'features' collection
        mongo_data = list(collection.find({}, {'_id': 0}))  # retrieves all documents without '_id'
        features_df = pd.DataFrame(mongo_data)  # converts list of dictionaries to DataFrame
        features_df['track_id'] = features_df['audio_file'].apply(lambda x: x.split('/')[-1].split('.')[0])  # extracts 'track_id' from 'audio_file' path
        return features_df  # returns DataFrame with features
    except Exception as e:
        print(f"Failed to load data from MongoDB: {e}")  
        return pd.DataFrame()  

def main():
    try:
        metadata_df = pd.read_csv('/home/mary/Downloads/metadata.csv')  # loads metadata from CSV file into DataFrame

        features_df = load_audio_features()  # loads audio features from MongoDB

        if not features_df.empty:  # checks if features DataFrame is not empty
            combined_df = pd.merge(metadata_df, features_df, on='track_id', how='left')  # merges metadata with features on 'track_id'
            
            if combined_df['track_id'].isna().any():
                print("Warning: Some audio features were not matched with metadata.")  # prints warning if any 'track_id' unmatched

            print(combined_df.head())  # prints first few rows of the merged DataFrame

        else:
            print("No features data to merge.") 
    except FileNotFoundError as fe:
        print(f"File not found: {fe}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()  
