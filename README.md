# Streamline Music Experience: A Spotify Alternative

### DS2004 - Fundamental of Big Data Analytics
## Introduction
This project aims to develop a streamlined music streaming service similar to Spotify, focusing on delivering a powerful music recommendation system. Utilizing big data analytics, machine learning, and web technologies, our system analyzes user preferences to suggest tracks tailored to their tastes, enhancing the overall listening experience.

## Dataset
We used the Free Music Archive (FMA) dataset, which includes a wide range of musical tracks and their metadata, aiding in diverse and accurate music recommendations.

## Data Loading and Pre-processing
### Load_Data.py
Purpose: Initializes a Spark session to efficiently load large sets of audio files for processing.

Key Functions:
spark.read.format("binaryFile").load("file:///media/mary/New Volume/sample_audio/")
df.show()
spark.stop()
spark.read.format("binaryFile").load("path_to_files"): Loads binary audio files into a Spark DataFrame.
df.show(): Displays the content of the DataFrame.

## Data Integration and MongoDB Usage
### merge.py
Purpose: Merges audio features extracted from the data with metadata to create a comprehensive dataset.

Key Functions:
MongoClient('mongodb://localhost:27017/')
pd.merge()
MongoClient: Connects to the local MongoDB database.
pd.merge(): Merges feature data with metadata ensuring all information is comprehensively integrated.

## Feature Extraction
### feature_extraction.py
Purpose: Extracts essential audio features using librosa for analyzing and recommending music.

Key Functions:
librosa.load(), librosa.feature.mfcc(), librosa.feature.spectral_centroid(), librosa.feature.zero_crossing_rate()
These functions extract MFCCs, spectral centroid, and zero-crossing rate, which are critical for understanding and processing audio data.


## Model Training and Recommendations
### training_model.py
Purpose: Trains a neural network to learn song features and recommend songs based on user preferences.

Key Functions:
torch.utils.data.Dataset, torch.nn.Module
train_embedding_model(), recommend_songs()
Custom PyTorch dataset and model classes are used for handling song features.
train_embedding_model(): Trains the model using song features.
recommend_songs(): Generates song recommendations based on the trained model.

## Challenges and Solutions
Data Scalability: Managing the large size of the FMA dataset was challenging. Utilizing Spark and MongoDB allowed efficient handling and processing of large datasets.

Feature Selection: Determining which features were most effective for recommendations required extensive testing and validation.


## Conclusion
This project successfully demonstrates the capability to build a music recommendation system. Future work could explore the integration of more complex algorithms and real-time data processing to enhance recommendation accuracy.

## Credits
[Amna Javaid]

[Maryam Khalid]

[Tabidah Usmani]
