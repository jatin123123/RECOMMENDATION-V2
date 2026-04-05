"""
Simple Music Recommendation System - Recommender Module
This module provides the recommendation logic.

Key Classes:
- MusicRecommender: Main class that loads models and generates recommendations

How Hybrid Recommendations Work:
1. Content-based: Find songs with similar audio features
2. Collaborative: Find songs liked by users with similar taste
3. Hybrid: Combine both scores with weighted average
"""

import pandas as pd
import numpy as np
from scipy import sparse
import joblib
import os


class MusicRecommender:
    """
    Simple Hybrid Music Recommender
    
    This class combines:
    - Content-based filtering (similar audio features)
    - Collaborative filtering (users who liked this also liked...)
    """
    
    def __init__(self, model_dir="models"):
        """Load all trained models"""
        self.model_dir = model_dir
        
        # Check if models exist
        if not os.path.exists(f"{model_dir}/nn_model.pkl"):
            raise FileNotFoundError("Models not found! Run simple_train.py first.")
        
        print("Loading models...")
        
        # Load content-based model components
        self.content_matrix = sparse.load_npz(f"{model_dir}/content_matrix.npz")
        self.nn_model = joblib.load(f"{model_dir}/nn_model.pkl")
        
        # Load track data
        self.tracks = pd.read_csv(f"{model_dir}/tracks.csv")
        self.tracks["track_id"] = self.tracks["track_id"].astype(str)
        
        # Create track_id to index mapping
        self.track_to_idx = {tid: i for i, tid in enumerate(self.tracks["track_id"])}
        self.idx_to_track = {i: tid for tid, i in self.track_to_idx.items()}
        
        # Load collaborative filtering model (if available)
        self.cf_available = False
        self.item_embeddings = None
        self.cf_track_to_idx = {}
        self.cf_idx_to_track = {}
        
        if os.path.exists(f"{model_dir}/item_embeddings.npy"):
            self.item_embeddings = np.load(f"{model_dir}/item_embeddings.npy")
            self.cf_track_to_idx = joblib.load(f"{model_dir}/track_to_idx.pkl")
            self.cf_idx_to_track = joblib.load(f"{model_dir}/idx_to_track.pkl")
            self.cf_available = True
            print("Collaborative filtering: ENABLED")
        else:
            print("Collaborative filtering: DISABLED (no user data)")
        
        print(f"Loaded {len(self.tracks)} tracks")
    
    def search_tracks(self, query, limit=10):
        """
        Search for tracks by name or artist
        Uses simple string matching
        """
        query = query.lower()
        
        results = []
        for idx, row in self.tracks.iterrows():
            track_name = str(row.get("track_name", "")).lower()
            artist_name = str(row.get("artist_name", "")).lower()
            
            # Check if query matches track or artist
            if query in track_name or query in artist_name:
                score = 1.0 if query == track_name else 0.8
                results.append({
                    "track_id": row["track_id"],
                    "track_name": row.get("track_name", "Unknown"),
                    "artist_name": row.get("artist_name", "Unknown"),
                    "score": score
                })
        
        # Sort by score and return top results
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        return results[:limit]
    
    def get_content_recommendations(self, track_id, n_recommendations=20):
        """
        Get recommendations based on audio features (content-based)
        
        How it works:
        1. Find the track's index
        2. Use Nearest Neighbors to find similar tracks
        3. Convert distances to similarity scores
        """
        if track_id not in self.track_to_idx:
            return {}
        
        idx = self.track_to_idx[track_id]
        
        # Find nearest neighbors
        distances, indices = self.nn_model.kneighbors(
            self.content_matrix[idx],
            n_neighbors=min(n_recommendations + 1, len(self.tracks))
        )
        
        # Convert to similarity scores (1 - distance)
        scores = {}
        for dist, neighbor_idx in zip(distances[0], indices[0]):
            neighbor_id = self.idx_to_track[neighbor_idx]
            if neighbor_id != track_id:  # Don't include the query track
                scores[neighbor_id] = 1 - dist  # Convert distance to similarity
        
        return scores
    
    def get_collaborative_recommendations(self, track_id, n_recommendations=20):
        """
        Get recommendations based on user behavior (collaborative filtering)
        
        How it works:
        1. Get the track's embedding vector
        2. Find other tracks with similar embeddings
        3. Use cosine similarity
        """
        if not self.cf_available or track_id not in self.cf_track_to_idx:
            return {}
        
        idx = self.cf_track_to_idx[track_id]
        track_vector = self.item_embeddings[idx]
        
        # Calculate cosine similarity with all tracks
        # Cosine similarity = (A · B) / (||A|| * ||B||)
        norms = np.linalg.norm(self.item_embeddings, axis=1)
        norms[norms == 0] = 1  # Avoid division by zero
        
        query_norm = np.linalg.norm(track_vector)
        if query_norm == 0:
            return {}
        
        similarities = np.dot(self.item_embeddings, track_vector) / (norms * query_norm)
        
        # Get top similar tracks
        scores = {}
        top_indices = np.argsort(similarities)[::-1][:n_recommendations + 1]
        
        for neighbor_idx in top_indices:
            neighbor_id = self.cf_idx_to_track.get(neighbor_idx)
            if neighbor_id and neighbor_id != track_id:
                scores[neighbor_id] = float(similarities[neighbor_idx])
        
        return scores
    
    def get_hybrid_recommendations(self, track_id, n_recommendations=20, content_weight=0.7):
        """
        Get hybrid recommendations combining both methods
        
        Formula: hybrid_score = content_weight * content_score + (1 - content_weight) * cf_score
        
        Parameters:
        - track_id: The seed track
        - n_recommendations: How many songs to recommend
        - content_weight: Balance between content (0-1), default 0.7
        """
        # Get scores from both methods
        content_scores = self.get_content_recommendations(track_id, n_recommendations * 2)
        cf_scores = self.get_collaborative_recommendations(track_id, n_recommendations * 2)
        
        # Combine all track IDs
        all_tracks = set(content_scores.keys()) | set(cf_scores.keys())
        
        if not all_tracks:
            return []
        
        # Normalize scores to 0-1 range
        def normalize(scores):
            if not scores:
                return {}
            values = list(scores.values())
            min_val, max_val = min(values), max(values)
            if max_val - min_val < 0.0001:
                return {k: 0.5 for k in scores}
            return {k: (v - min_val) / (max_val - min_val) for k, v in scores.items()}
        
        content_norm = normalize(content_scores)
        cf_norm = normalize(cf_scores)
        
        # Calculate hybrid scores
        recommendations = []
        for tid in all_tracks:
            content_score = content_norm.get(tid, 0)
            cf_score = cf_norm.get(tid, 0)
            
            # Weighted combination
            hybrid_score = content_weight * content_score + (1 - content_weight) * cf_score
            
            # Get track info
            track_info = self.tracks[self.tracks["track_id"] == tid].iloc[0] if tid in self.track_to_idx else None
            
            if track_info is not None:
                recommendations.append({
                    "track_id": tid,
                    "track_name": track_info.get("track_name", "Unknown"),
                    "artist_name": track_info.get("artist_name", "Unknown"),
                    "preview_url": track_info.get("preview_url", ""),
                    "content_score": content_score,
                    "cf_score": cf_score if self.cf_available else None,
                    "hybrid_score": hybrid_score
                })
        
        # Sort by hybrid score
        recommendations.sort(key=lambda x: x["hybrid_score"], reverse=True)
        
        return recommendations[:n_recommendations]


# Simple test
if __name__ == "__main__":
    recommender = MusicRecommender()
    print("\nSearching for 'love'...")
    results = recommender.search_tracks("love", limit=5)
    for r in results:
        print(f"  - {r['track_name']} by {r['artist_name']}")
