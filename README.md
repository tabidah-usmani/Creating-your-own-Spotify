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
