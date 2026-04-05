"""
Simple Music Recommendation System - Streamlit Web App
A clean and simple web interface for music recommendations.

Run with: streamlit run simple_app.py
"""

import streamlit as st
import pandas as pd
from recommender import MusicRecommender

# Page config
st.set_page_config(
    page_title="🎵 Music Recommender",
    layout="wide"
)

# Title
st.title("🎵 Spotify Hybrid Music Recommendation System")
st.markdown("**Find similar songs using AI-powered hybrid recommendations**")

# Load the recommender (cached so it only loads once)
@st.cache_resource
def load_recommender():
    return MusicRecommender()


# Try to load the model
try:
    recommender = load_recommender()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    st.error("⚠️ Models not found! Please run `python train.py` first.")
    st.stop()


# Show what mode we're in
if recommender.cf_available:
    st.success("✅ **Hybrid Mode**: Using both Content-based + Collaborative Filtering")
else:
    st.warning("⚠️ **Content-only Mode**: No user data available for collaborative filtering")


# Sidebar - Settings
st.sidebar.header("⚙️ Settings")
playlist_size = st.sidebar.slider(
    "Number of recommendations",
    min_value=5,
    max_value=30,
    value=10
)

if recommender.cf_available:
    content_weight = st.sidebar.slider(
        "Content vs Collaborative weight",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        help="1.0 = only content-based, 0.0 = only collaborative"
    )
else:
    content_weight = 1.0

st.sidebar.text("@Jatin Jangid")

# How it works section
with st.expander("🧠 How does this work?"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Content-Based Filtering
        - Looks at **audio features** like energy, tempo, danceability
        - Uses **TF-IDF** on genre tags
        - Finds songs that **sound similar**
        - Works for any song in database
        """)
    
    with col2:
        st.markdown("""
        ### 👥 Collaborative Filtering
        - Uses **SVD** (matrix factorization)
        - Analyzes what songs users play together
        - Finds songs liked by **similar users**
        - Discovers unexpected recommendations
        """)
    
    st.markdown("""
    ### 🔀 Hybrid Approach
    ```
    hybrid_score = content_weight × content_score + (1 - content_weight) × cf_score
    ```
    Combines both methods for better recommendations!
    """)


# Main section - Search
st.subheader("🔍 Search for a song")
search_query = st.text_input(
    "Enter track name or artist",
    placeholder="e.g., Bohemian Rhapsody, Queen, Shape of You..."
)

if search_query:
    # Search for tracks
    search_results = recommender.search_tracks(search_query, limit=10)
    
    if search_results:
        # Create selection dropdown
        options = [f"{r['track_name']} - {r['artist_name']}" for r in search_results]
        selected = st.selectbox("Select a track:", options)
        
        # Get the selected track ID
        selected_idx = options.index(selected)
        selected_track = search_results[selected_idx]
        
        # Generate button
        if st.button("🎯 Get Recommendations", type="primary"):
            
            with st.spinner("Finding similar songs..."):
                recommendations = recommender.get_hybrid_recommendations(
                    selected_track["track_id"],
                    n_recommendations=playlist_size,
                    content_weight=content_weight
                )
            
            if recommendations:
                st.success(f"✨ Found {len(recommendations)} recommendations!")
                
                # Display recommendations
                st.subheader("🎵 Recommended Songs")
                
                for i, rec in enumerate(recommendations, 1):
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            st.markdown(f"**{i}. {rec['track_name']}**")
                            st.caption(f"Artist: {rec['artist_name']}")
                            
                            # Audio preview if available
                            if rec.get("preview_url") and str(rec["preview_url"]).startswith("http"):
                                st.audio(rec["preview_url"])
                        
                        with col2:
                            st.metric("Score", f"{rec['hybrid_score']:.2f}")
                            
                            # Show breakdown
                            score_text = f"Content: {rec['content_score']:.2f}"
                            if rec['cf_score'] is not None:
                                score_text += f" | CF: {rec['cf_score']:.2f}"
                            st.caption(score_text)
                        
                        st.divider()
                
                # Export option
                st.subheader("💾 Export")
                df = pd.DataFrame(recommendations)
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Download as CSV",
                    csv,
                    "recommendations.csv",
                    "text/csv"
                )
            else:
                st.warning("No recommendations found. Try a different song.")
    else:
        st.info("No tracks found. Try a different search term.")


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🎵 Music Recommendation System</p>
    <p>Built By Jatin Jangid</p>
</div>
""", unsafe_allow_html=True)
