"""
Simple Music Recommendation System - Training Script
This script trains both content-based and collaborative filtering models.

How it works:
1. Load music data (track info with audio features)
2. Build content-based model using audio features + TF-IDF on tags
3. Build collaborative filtering model using SVD on user listening history
4. Save trained models for later use
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from scipy import sparse
import joblib
import os

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)


def load_music_data():
    """Load and prepare music dataset"""
    print("Loading music data...")
    
    # Load the music info CSV
    df = pd.read_csv("data/Music Info.csv")
    
    # Rename columns to standard names if needed
    if "name" in df.columns:
        df["track_name"] = df["name"]
    if "artist" in df.columns:
        df["artist_name"] = df["artist"]
    if "tags" in df.columns:
        df["tags_text"] = df["tags"].fillna("")
    if "spotify_preview_url" in df.columns:
        df["preview_url"] = df["spotify_preview_url"]
    
    # Make sure track_id is string
    df["track_id"] = df["track_id"].astype(str)
    
    # Remove duplicates
    df = df.drop_duplicates(subset="track_id")
    
    print(f"Loaded {len(df)} tracks")
    return df


def train_content_model(df):
    """
    Train content-based filtering model
    
    Steps:
    1. Extract audio features (energy, tempo, valence, etc.)
    2. Normalize the features using StandardScaler
    3. Create TF-IDF vectors from genre/tag text
    4. Combine both into one feature matrix
    5. Train Nearest Neighbors model to find similar songs
    """
    print("\n--- Training Content-Based Model ---")
    
    # Audio features we want to use
    audio_features = ["energy", "valence", "tempo", "danceability", 
                      "acousticness", "instrumentalness", "loudness"]
    
    # Keep only features that exist in our data
    available_features = [f for f in audio_features if f in df.columns]
    print(f"Using audio features: {available_features}")
    
    # Step 1: Prepare numeric features
    numeric_data = df[available_features].copy()
    numeric_data = numeric_data.fillna(numeric_data.median())  # Fill missing with median
    
    # Step 2: Scale features to 0-1 range
    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(numeric_data)
    
    # Step 3: Create TF-IDF from tags/genres
    print("Creating TF-IDF vectors from tags...")
    tags = df["tags_text"].fillna("") if "tags_text" in df.columns else pd.Series([""] * len(df))
    
    tfidf = TfidfVectorizer(max_features=1000)
    text_matrix = tfidf.fit_transform(tags)
    
    # Step 4: Combine numeric and text features
    numeric_sparse = sparse.csr_matrix(numeric_scaled)
    content_matrix = sparse.hstack([numeric_sparse, text_matrix])
    
    print(f"Content matrix shape: {content_matrix.shape}")
    
    # Step 5: Train Nearest Neighbors model
    print("Training Nearest Neighbors model...")
    nn_model = NearestNeighbors(metric="cosine", algorithm="brute")
    nn_model.fit(content_matrix)
    
    # Save everything
    print("Saving content model...")
    sparse.save_npz("models/content_matrix.npz", content_matrix)
    joblib.dump(nn_model, "models/nn_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    joblib.dump(tfidf, "models/tfidf.pkl")
    joblib.dump(available_features, "models/feature_names.pkl")
    
    # Save track metadata
    df.to_csv("models/tracks.csv", index=False)
    
    print("Content-based model saved!")
    return content_matrix, nn_model


def train_collaborative_model(df):
    """
    Train collaborative filtering model using SVD
    
    Steps:
    1. Load user listening history
    2. Create user-item matrix
    3. Apply SVD to find latent factors
    4. Save item embeddings for similarity search
    """
    print("\n--- Training Collaborative Filtering Model ---")
    
    # Check if user history file exists
    history_path = "data/User Listening History.csv"
    if not os.path.exists(history_path):
        print("No user history file found. Skipping collaborative filtering.")
        return None
    
    # Load user listening history
    print("Loading user listening history...")
    history = pd.read_csv(history_path)
    history["track_id"] = history["track_id"].astype(str)
    
    # Keep only tracks that exist in our music data
    valid_tracks = set(df["track_id"])
    history = history[history["track_id"].isin(valid_tracks)]
    
    if len(history) == 0:
        print("No matching tracks found. Skipping collaborative filtering.")
        return None
    
    print(f"Using {len(history)} listening records")
    
    # Create mappings
    user_ids = history["user_id"].unique()
    track_ids = history["track_id"].unique()
    
    user_to_idx = {uid: i for i, uid in enumerate(user_ids)}
    track_to_idx = {tid: i for i, tid in enumerate(track_ids)}
    idx_to_track = {i: tid for tid, i in track_to_idx.items()}
    
    # Create user-item matrix
    print("Creating user-item matrix...")
    rows = history["user_id"].map(user_to_idx).values
    cols = history["track_id"].map(track_to_idx).values
    
    # Use play count as weight (or 1 if not available)
    if "playcount" in history.columns:
        weights = np.log1p(history["playcount"].fillna(1).values)  # Log transform
    else:
        weights = np.ones(len(history))
    
    user_item_matrix = sparse.csr_matrix(
        (weights, (rows, cols)), 
        shape=(len(user_ids), len(track_ids))
    )
    
    print(f"User-item matrix: {user_item_matrix.shape}")
    
    # Apply SVD (like matrix factorization)
    print("Applying SVD...")
    n_factors = min(50, min(user_item_matrix.shape) - 1)
    svd = TruncatedSVD(n_components=n_factors, random_state=42)
    svd.fit(user_item_matrix)
    
    # Get item embeddings (tracks represented as vectors)
    item_embeddings = svd.components_.T  # Shape: (n_tracks, n_factors)
    
    # Save collaborative model
    print("Saving collaborative model...")
    np.save("models/item_embeddings.npy", item_embeddings)
    joblib.dump(track_to_idx, "models/track_to_idx.pkl")
    joblib.dump(idx_to_track, "models/idx_to_track.pkl")
    
    print("Collaborative filtering model saved!")
    return item_embeddings


if __name__ == "__main__":
    # Load data
    df = load_music_data()
    
    # Train both models
    train_content_model(df)
    train_collaborative_model(df)
    
    print("\n=== Training Complete! ===")
    print("Models saved in 'models/' folder")
    print("Run 'streamlit run simple_app.py' to start the app")
