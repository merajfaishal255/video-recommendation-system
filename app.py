# Author: merajfaishal255
# Project: Local Video Recommendation System

import streamlit as st
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Video Player", page_icon="🎬")
st.title("🎬 Local Video Recommender")
st.sidebar.markdown("Author: [merajfaishal255](https://github.com/merajfaishal255)")

# 1. Load Data
try:
    df = pd.read_csv("data.csv")
except FileNotFoundError:
    st.error("❌ 'data.csv' not found. Please create it first.")
    st.stop()

# 2. Logic (Cosine Similarity)
cv = CountVectorizer()
try:
    count_matrix = cv.fit_transform(df['tags'])
    cosine_sim = cosine_similarity(count_matrix)
except ValueError:
    st.error("❌ Your CSV 'tags' column is empty or missing.")
    st.stop()

def recommend(video_title):
    try:
        idx = df[df['title'] == video_title].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:4] # Top 3 similar
        
        video_indices = [i[0] for i in sim_scores]
        return df.iloc[video_indices]
    except:
        return pd.DataFrame()

# 3. User Interface
selected_video = st.selectbox("Select a video to watch:", df['title'].values)

if st.button("Play & Recommend"):
    
    # --- PART A: Play the Selected Video ---
    row = df[df['title'] == selected_video].iloc[0]
    file_name = row['filename']
    
    # Look for the file in 'videos' folder
    file_path = f"videos/{file_name}.mp4"
    
    st.subheader(f"▶️ Now Playing: {selected_video}")
    
    if os.path.exists(file_path):
        video_file = open(file_path, 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)
    else:
        st.error(f"❌ File not found: {file_path}")
        st.info("Check your 'videos' folder and filename spelling.")

    # --- PART B: Show Recommendations ---
    st.markdown("---")
    st.subheader("🧐 Recommended Next:")
    
    recs = recommend(selected_video)
    
    cols = st.columns(3)
    for idx, (col, row) in enumerate(zip(cols, recs.iterrows())):
        rec_data = row[1]
        rec_filename = rec_data['filename']
        rec_path = f"videos/{rec_filename}.mp4"
        
        with col:
            st.write(f"**{rec_data['title']}**")
            # We try to show a preview if the file exists
            if os.path.exists(rec_path):
                st.video(rec_path)
            else:
                st.warning(f"File missing: {rec_filename}")