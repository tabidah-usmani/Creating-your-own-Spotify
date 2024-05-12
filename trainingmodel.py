import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pymongo import MongoClient

# Custom Dataset for Song Features from MongoDB
class SongFeaturesDataset(Dataset):
    def _init(self, features):  # Corrected from _init to _init_
        self.features = features

    def _len(self):  # Corrected from _len to _len_
        return len(self.features)

    def _getitem(self, idx):  # Corrected from _getitem to _getitem_
        return torch.tensor(self.features[idx], dtype=torch.float32)

# Neural Network Model for Learning Song Embeddings
class FeatureEmbeddingModel(nn.Module):
    def _init(self, input_dim, embedding_dim=64):  # Corrected from _init to _init_
        super(FeatureEmbeddingModel, self)._init()  # Corrected from _init to _init_
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, embedding_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

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

    # Assuming each document structure as described and mfcc, spectral_centroid, zero_crossing_rate are stored as lists
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

# Main function
def main():
    features = get_features_from_mongodb()
    if features.size == 0:
        print("No features to train on. Exiting...")
        return
    print(f"Features shape: {features.shape}")
    model = train_embedding_model(features, epochs=10)
    song_index = 1  # Adjust based on your data
    recommended_ids = recommend_songs(model, features, song_index)
    print("Recommended Song IDs:", recommended_ids)


if _name_ == "_main":  # Corrected from _name to _name_ and main to _main_
    main()
