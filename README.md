# 🎵 Streamline Music Experience: A Spotify Alternative

**DS2004 — Fundamentals of Big Data Analytics**

A end-to-end music streaming and recommendation platform built as a big-data analytics project. The system ingests raw audio, extracts acoustic features at scale, learns song embeddings with a neural network, and serves personalized recommendations through a simple web app — mirroring the core pipeline behind services like Spotify.

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Pipeline Walkthrough](#pipeline-walkthrough)
  - [1. Data Loading](#1-data-loading--load_datapy)
  - [2. Feature Extraction](#2-feature-extraction--feature_extractionpy)
  - [3. Data Integration](#3-data-integration--mergepy)
  - [4. Model Training & Recommendations](#4-model-training--recommendations--training_modelpy)
  - [5. Deployment](#5-deployment--apppy)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Challenges and Solutions](#challenges-and-solutions)
- [Results](#results)
- [Future Work](#future-work)
- [Credits](#credits)
- [License](#license)

---

## Overview

This project simulates a real-world music streaming pipeline end-to-end:

1. **Ingest** thousands of raw audio files efficiently using Apache Spark.
2. **Extract** meaningful acoustic features (timbre, pitch, rhythm) from each track using `librosa`.
3. **Integrate** those features with rich track metadata using MongoDB and Pandas.
4. **Train** a neural embedding model in PyTorch that learns a latent representation of each song.
5. **Recommend** similar songs to users based on embedding similarity.
6. **Deploy** the whole system behind a lightweight web app for streaming and recommendations.

The goal was to explore how big data tools (Spark, MongoDB) and machine learning (PyTorch, librosa-based feature engineering) combine to power a content-based music recommendation engine.

## System Architecture

```
┌─────────────────┐     ┌──────────────────────┐     ┌───────────────────────┐
│   Raw Audio     │ --> │   Spark Data Loader  │ --> │  Feature Extractor    │
│  (FMA Dataset)  │     │   (Load_Data.py)     │     │(feature_extraction.py)│
└─────────────────┘     └──────────────────────┘     └─────────┬─────────────┘
                                                               │
                                                               v
┌─────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
│   Metadata      │ --> │   MongoDB + Merge    │ <-- │  Extracted Features  │
│  (FMA metadata) │     │   (merge.py)         │     │  (MFCC, centroid...) │
└─────────────────┘     └──────────┬───────────┘     └──────────────────────┘
                                   │
                                   v
                          ┌──────────────────────────┐
                          │  Embedding Model         │
                          │  (training_model.py)     │
                          │  PyTorch Neural Network  │
                          └──────────┬───────────────┘
                                     │
                                     v
                          ┌──────────────────────────────┐
                          │  Web App / Deployment        │
                          │  (app.py)                    │
                          │  Recommendations + Streaming │
                          └──────────────────────────────┘
```

## Dataset

We use the **[Free Music Archive (FMA)](https://github.com/mdeff/fma)** dataset, an open, curated collection of full-length audio tracks paired with rich metadata (genre, artist, album, tags, etc.). FMA was chosen because it:

- Provides free, legally usable audio for research and experimentation
- Spans a wide range of genres, encouraging diverse and generalizable recommendations
- Includes pre-computed metadata that pairs well with custom-extracted audio features

## Tech Stack

| Layer | Technology |
|---|---|
| Distributed data loading | Apache Spark (PySpark) |
| Audio feature extraction | librosa |
| Metadata storage & integration | MongoDB, Pandas |
| Model training | PyTorch |
| Web deployment | Python (Flask/similar — see `app.py`) |
| Language | Python 3 |

## Project Structure

```
.
├── Load_Data.py            # Spark-based audio ingestion
├── feature_extraction.py   # Audio feature extraction (MFCC, spectral centroid, ZCR)
├── merge.py                 # Merges features + metadata via MongoDB/Pandas
├── training_model.py        # PyTorch dataset, model, training & recommendation logic
├── app.py                   # Web app for serving recommendations and streaming
└── README.md
```

## Pipeline Walkthrough

### 1. Data Loading — `Load_Data.py`

Initializes a Spark session and loads large batches of raw audio files into a distributed DataFrame for efficient downstream processing.

**Key functions:**

```python
spark.read.format("binaryFile").load("file:///media/mary/New Volume/sample_audio/")
df.show()
spark.stop()
```

- `spark.read.format("binaryFile").load(path)` — Loads binary audio files into a Spark DataFrame, allowing large volumes of audio to be handled without loading everything into local memory at once.
- `df.show()` — Displays a preview of the loaded DataFrame for verification.
- `spark.stop()` — Gracefully shuts down the Spark session once loading is complete.

> ⚠️ Note: the file path is currently hardcoded to a local mount (`/media/mary/New Volume/sample_audio/`). Update this to your own audio directory before running.

### 2. Feature Extraction — `feature_extraction.py`

Extracts the acoustic features that describe each track's timbre, tonal color, and rhythmic texture — the building blocks the recommendation model learns from.

**Key functions:**

```python
librosa.load()
librosa.feature.mfcc()
librosa.feature.spectral_centroid()
librosa.feature.zero_crossing_rate()
```

- `librosa.load()` — Loads an audio file as a waveform array with its sampling rate.
- `librosa.feature.mfcc()` — Computes Mel-Frequency Cepstral Coefficients, capturing the timbral texture of the audio.
- `librosa.feature.spectral_centroid()` — Measures the "center of mass" of the spectrum, indicating the brightness of a sound.
- `librosa.feature.zero_crossing_rate()` — Measures how frequently the signal changes sign, useful for distinguishing percussive vs. tonal content.

Together, these features form a compact numerical fingerprint of each track.

### 3. Data Integration — `merge.py`

Combines the extracted audio features with each track's metadata (title, artist, genre, etc.) into a single, unified dataset ready for model training.

**Key functions:**

```python
MongoClient('mongodb://localhost:27017/')
pd.merge()
```

- `MongoClient('mongodb://localhost:27017/')` — Connects to a local MongoDB instance where metadata is stored.
- `pd.merge()` — Joins the audio feature table with the metadata table (typically on a track/file ID), producing a single comprehensive dataset.

### 4. Model Training & Recommendations — `training_model.py`

Trains a neural embedding model that maps each song's feature vector into a latent space where musically similar songs sit close together, then uses that space to generate recommendations.

**Key components:**

```python
torch.utils.data.Dataset
torch.nn.Module
train_embedding_model()
recommend_songs()
```

- A custom `torch.utils.data.Dataset` subclass wraps the merged feature/metadata table for efficient batching.
- A custom `torch.nn.Module` subclass defines the embedding network architecture.
- `train_embedding_model()` — Runs the training loop, optimizing the network to produce meaningful song embeddings.
- `recommend_songs()` — Given a seed song (or user preference vector), retrieves the nearest songs in embedding space as recommendations.

### 5. Deployment — `app.py`

Serves the trained recommendation system through a web application, allowing users to browse recommendations and stream tracks.

**Key functions:**

- `recommendations()` — Endpoint/handler that returns personalized song recommendations.
- `streaming()` — Endpoint/handler that streams the selected audio track to the user.

## Getting Started

### Prerequisites

- Python 3.8+
- Apache Spark (PySpark)
- MongoDB (running locally or accessible remotely)
- Python packages: `librosa`, `pandas`, `pymongo`, `torch`, `pyspark`, and your chosen web framework (e.g., `flask`)

### Installation

```bash
git clone https://github.com/<your-username>/streamline-music-experience.git
cd streamline-music-experience
pip install -r requirements.txt
```

> If a `requirements.txt` doesn't exist yet, consider adding one listing `pyspark`, `librosa`, `pandas`, `pymongo`, `torch`, and your web framework, pinned to the versions you used.

### Configuration

1. Update the audio file path in `Load_Data.py` to point to your local dataset directory.
2. Ensure MongoDB is running locally (`mongodb://localhost:27017/`) or update the connection string in `merge.py`.
3. Download and extract the [FMA dataset](https://github.com/mdeff/fma) subset you intend to use.

## Usage

Run the pipeline stages in order:

```bash
# 1. Load raw audio into Spark
python Load_Data.py

# 2. Extract audio features
python feature_extraction.py

# 3. Merge features with metadata
python merge.py

# 4. Train the recommendation model
python training_model.py

# 5. Launch the web app
python app.py
```

Then open the app in your browser to explore recommendations and stream tracks.

## Challenges and Solutions

| Challenge | Solution |
|---|---|
| **Data scalability** — the FMA dataset is large, and loading/processing it naively didn't scale | Used Apache Spark for distributed loading and MongoDB for flexible, queryable metadata storage |
| **Feature selection** — determining which audio features actually improved recommendation quality | Ran extensive testing and validation across MFCC, spectral centroid, and zero-crossing rate combinations to identify the most informative feature set |

## Future Work

- Integrate more sophisticated recommendation algorithms (e.g., collaborative filtering, hybrid models)
- Support real-time/streaming data processing instead of batch pipelines
- Add user-facing feedback loops (likes/skips) to personalize recommendations over time
- Expand deployment with authentication, playlists, and a richer front-end

## Credits

- **Amna Javaid**
- **Maryam Khalid**
- **Tabidah Usmani**

Developed as part of the **DS2004 – Fundamentals of Big Data Analytics** course.
