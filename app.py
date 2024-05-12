from flask import Flask, render_template, Response
from torch.utils.data import Dataset, DataLoader
from model import FeatureEmbeddingModel
import torch
import numpy as np
from pymongo import MongoClient
from pydub import AudioSegment
import os

app = Flask(__name__, static_folder='static')

# Custom Dataset for Song Features from MongoDB
class SongFeaturesDataset(Dataset):
    def __init__(self, features):  
        self.features = features

    def __len__(self):  
        return len(self.features)

    def __getitem__(self, idx):  
        return torch.tensor(self.features[idx], dtype=torch.float32)

# Connect to MongoDB and fetch features
def get_features_from_mongodb():
    client = MongoClient('localhost', 27017)
    db = client['audio_features']
    collection = db['features']
    data = list(collection.find({}, {'_id': 0, 'mfcc': 1, 'spectral_centroid': 1, 'zero_crossing_rate': 1}))
    if not data:
        print("No data retrieved from MongoDB.")
    else:
        print(f"Retrieved {len(data)} records.")

    features = [np.concatenate([item['mfcc'], item['spectral_centroid'], item['zero_crossing_rate']])
                for item in data if 'mfcc' in item and 'spectral_centroid' in item and 'zero_crossing_rate' in item]
    
    return np.array(features)

# Train the Embedding Model
def train_embedding_model(features, epochs=10, batch_size=16):
    dataset = SongFeaturesDataset(features)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = FeatureEmbeddingModel(input_dim=features.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()
    for epoch in range(epochs):
        for data in dataloader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, data)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    return model

# Recommend Songs
def recommend_songs(model, features, song_index, top_n=5):
    model.eval()
    with torch.no_grad():
        embeddings = model(torch.tensor(features, dtype=torch.float32)).numpy()
    song_embedding = embeddings[song_index]
    similarities = np.dot(embeddings, song_embedding)
    most_similar_ids = np.argsort(similarities)[::-1][1:top_n+1]
    return most_similar_ids

# Function to fetch and stream music
def fetch_and_stream_music(song_id):
    audio_file_path = f"/home/asmariaz/Downloads/amna-project/sampled_audio_1gb/{song_id}.mp3"  # Update the path
    try:
        audio = AudioSegment.from_file(audio_file_path)
    except FileNotFoundError:
        return Response("File not found", status=404)

    raw_audio_data = audio.raw_data
    mimetype = 'audio/mpeg'

    # Return audio data as a Flask Response
    return Response(raw_audio_data, mimetype=mimetype)

# Main function
def main():
    features = get_features_from_mongodb()
    if features.size == 0:
        print("No features to train on. Exiting...")
        return
    print(f"Features shape: {features.shape}")
    model = train_embedding_model(features, epochs=10)
    song_index = 1  
    recommended_ids = recommend_songs(model, features, song_index)
    print("Recommended Song IDs:", recommended_ids)

@app.route('/recommendations/<user_id>')
def recommendations(user_id):
    main()
    client = MongoClient('localhost', 27017)
    db = client['audio_feature']
    collection = db['features']

    # Query MongoDB to fetch recommendations for the given user_id
    user_recommendations = collection.find_one({'user_id': user_id}, {'_id': 0, 'recommendations': 1})

    # Extract recommendations from the query result
    if user_recommendations:
        return user_recommendations.get('recommendations', [])  # Return recommendations if found
    else:
        return []  # Return an empty list if no recommendations found

# Route for the streaming page
@app.route('/streaming/<song_id>')
def streaming(song_id):
    # Logic to fetch and stream music
    audio_response = fetch_and_stream_music(song_id)
    return audio_response

if __name__ == '__main__':
    app.run(debug=True)
