import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 1. Load The Arcane Artifacts ---
# Load the k-NN model, scaler, and the processed data
try:
    knn_model = joblib.load('knn_model.joblib')
    scaler = joblib.load('scaler.joblib')
    df = pd.read_pickle('processed_data.pkl')
    # Combine song name and artist for the search box
    df['song_artist'] = df['name'] + " by " + df['artists'].str.replace(r"[\[\]']", "", regex=True)
except FileNotFoundError:
    st.error("The model's artifacts are missing from the ether. Please run model_training.py to conjure them.")
    st.stop()

# --- 2. Conjure the Gothic UI ---
st.set_page_config(page_title="Nocturne AI", layout="wide", page_icon="🦇")

# Custom CSS for the gothic theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Crimson+Text:wght@400;700&display=swap');

html, body, [class*="st-"] {
    font-family: 'Crimson Text', serif;
}

.stApp {
    background-color: #1a1a1a;
    color: #e0e0e0;
}

h1, h2, h3 {
    color: #9a4ca6; /* Muted Purple */
}

.stButton>button {
    border: 2px solid #5c2c69;
    background-color: transparent;
    color: #e0e0e0;
    transition: all 0.3s ease-in-out;
}

.stButton>button:hover {
    border-color: #c789d6;
    background-color: #5c2c69;
    color: #ffffff;
}

.stButton>button:focus {
    box-shadow: 0 0 0 2px #c789d6;
    outline: none;
}

/* Style for primary button in Vibe Builder */
.stButton[data-testid="stFormSubmitButton"]>button, .stButton[aria-label="Generate My Vibe Playlist! ✨"]>button {
    background-color: #9a4ca6;
    color: #ffffff;
    border: 2px solid #9a4ca6;
}
.stButton[data-testid="stFormSubmitButton"]>button:hover, .stButton[aria-label="Generate My Vibe Playlist! ✨"]>button:hover {
     background-color: #5c2c69;
     border: 2px solid #c789d6;
}

[data-baseweb="tab-list"] {
    gap: 24px;
}

[data-baseweb="tab"] {
    height: 50px;
    padding-left: 20px;
    padding-right: 20px;
    background-color: transparent;
    border-radius: 4px;
}

[data-baseweb="tab"]:hover {
    background-color: #2a2a2a;
}

[data-baseweb="tab"][aria-selected="true"] {
    background-color: #5c2c69;
}
</style>
""", unsafe_allow_html=True)

st.title("🦇 Nocturne AI 🥀")
st.markdown("Uncover the hidden echoes in music. Choose your ritual below.")

# --- 3. The Two Rituals ---
tab1, tab2 = st.tabs(["**I. Echoes of a Melody**", "**II. Summon the Vibe**"])

# --- Ritual 1: Echoes of a Melody ---
with tab1:
    st.header("Find a Song's Ghostly Twin")
    
    selected_song = st.selectbox(
        'Search the archives for a known melody...',
        options=df['song_artist'].values,
        index=None,
        placeholder="Whisper a song's name..."
    )

    if st.button("Unveil Similar Melodies 🕯️", key='song_matcher'):
        if selected_song:
            # Find song's essence
            song_index = df[df['song_artist'] == selected_song].index[0]
            song_features = df.iloc[song_index][['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence']].values.reshape(1, -1)
            
            # Scale features and find its kin
            scaled_features = scaler.transform(song_features)
            distances, indices = knn_model.kneighbors(scaled_features)
            
            st.subheader(f"Spirits that resonate with '{selected_song}':")
            for i in range(1, len(indices[0])):
                recommended_song = df.iloc[indices[0][i]]
                st.write(f"**{i}. {recommended_song['name']}** by {recommended_song['artists'].replace(r'[\[\]\']', '', regex=True)}")
        else:
            st.warning("You must select a melody first.")

# --- Ritual 2: Summon the Vibe ---
with tab2:
    st.header("Weave a Playlist from Sheer Mood")
    st.markdown("Choose the fragments of sound that please you. We shall find others that share their soul.")

    # Initialize session state for chosen fragments
    if 'chosen_songs' not in st.session_state:
        st.session_state.chosen_songs = []
        st.session_state.song_pool = df.sample(9)

    # Display song tiles for selection
    cols = st.columns(3)
    for i, song in enumerate(st.session_state.song_pool.itertuples()):
        with cols[i % 3]:
            if st.button(f"{song.name} by {song.artists.replace(r'[\[\]\']', '', regex=True)}", key=song.id):
                if song.id not in [s['id'] for s in st.session_state.chosen_songs]:
                    st.session_state.chosen_songs.append({'name': song.name, 'artists': song.artists, 'id': song.id})
                    st.toast(f"'{song.name}' has been added to the summoning circle.")

    st.divider()

    col_a, col_b = st.columns([1,1])
    with col_a:
        if st.button("Reveal new fragments 💀"):
            st.session_state.song_pool = df.sample(9)
            st.rerun()

    with col_b:
        if st.button("Summon My Vibe Playlist! 🔮", key="generate_vibe"):
            if not st.session_state.chosen_songs:
                st.warning("The summoning circle is empty! Choose some songs to define the vibe.")
            else:
                chosen_ids = [s['id'] for s in st.session_state.chosen_songs]
                chosen_features = df[df['id'].isin(chosen_ids)][['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness', 'speechiness', 'tempo', 'valence']]
                average_vibe = chosen_features.mean().values.reshape(1, -1)
                
                scaled_vibe = scaler.transform(average_vibe)
                distances, indices = knn_model.kneighbors(scaled_vibe)

                st.subheader("Your custom incantation reveals these songs:")
                rec_count = 0
                for i in range(len(indices[0])):
                    if rec_count >= 10: break
                    rec_song_id = df.iloc[indices[0][i]]['id']
                    if rec_song_id not in chosen_ids:
                        rec_song_info = df.iloc[indices[0][i]]
                        st.write(f"**{rec_count+1}. {rec_song_info['name']}** by {rec_song_info['artists'].replace(r'[\[\]\']', '', regex=True)}")
                        rec_count += 1


    # Display chosen songs in a sidebar
    with st.sidebar:
        st.header("The Summoning Circle")
        if st.session_state.chosen_songs:
            for song in st.session_state.chosen_songs:
                st.write(f"- {song['name']}")
            if st.button("Banish All"):
                st.session_state.chosen_songs = []
                st.rerun()
        else:
            st.info("The circle is empty.")
