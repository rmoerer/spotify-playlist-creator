import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import os
import ast
import pandas as pd
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from gemini_stuff import make_search_model, make_playlist_model, make_tracks_model, sort_by_cosine_similarity
from spotify_stuff import get_playlists, get_tracks

# Load environment variables
load_dotenv()

# Spotify configuration
CLIENT_ID = os.getenv('CLIENT_ID')
CLIENT_SECRET = os.getenv('CLIENT_SECRET')
REDIRECT_URI = os.getenv('REDIRECT_URI')
SCOPE = 'playlist-read-private playlist-modify-public'

# Initialize Streamlit app
st.title('Spotify Playlist Creator')

# Spotify authentication
if 'token_info' not in st.session_state:
    sp_oauth = SpotifyOAuth(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, redirect_uri=REDIRECT_URI, scope=SCOPE)
    token_info = sp_oauth.get_cached_token()
    
    if not token_info:
        auth_url = sp_oauth.get_authorize_url()
        st.write(f"### [Click here to authorize with Spotify]({auth_url})")
        st.stop()
    else:
        st.session_state.token_info = token_info

# Handling redirect back from Spotify with the auth code
query_params = st.query_params

if 'code' in query_params:
    sp_oauth = SpotifyOAuth(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, redirect_uri=REDIRECT_URI, scope=SCOPE)
    code = query_params['code'][0]
    token_info = sp_oauth.get_access_token(code)
    st.session_state.token_info = token_info
    st.experimental_rerun()

# If token_info is in session state, proceed with Spotify API calls
if 'token_info' in st.session_state:
    token_info = st.session_state.token_info
    sp = spotipy.Spotify(auth=token_info['access_token'])

    # Display some information or functionality using Spotify API
    user_profile = sp.current_user()
    st.write(f"Logged in as {user_profile['display_name']}")

if 'tracks_displayed' not in st.session_state:
    st.session_state.tracks_displayed = False

# User input for playlist prompt
prompt = st.text_input('Enter a prompt for your playlist:')
if st.button('Generate Playlist') and prompt:

    # print to user that we are searching spotify
    message_placeholder = st.empty()
    message_placeholder.write('Searching spotify...')

    # Generate search strings for Spotify
    search_model = make_search_model()
    response = search_model.generate_content(prompt)
    search_strings = ast.literal_eval(response.text.strip())
    print(search_strings)

    # Search Spotify for playlists that spotify mixes
    playlists = get_playlists(sp, search_strings)
    playlists_df = pd.DataFrame(playlists)
    playlists_df = playlists_df.drop_duplicates(subset='playlist_id')

    # Convert prompt to phrase for semantic search
    playlist_model = make_playlist_model(playlists_df['playlist_name'].tolist())
    playlist_response = playlist_model.generate_content(prompt)
    playlist_prompt = response.text.strip()
    playlist_df = sort_by_cosine_similarity(playlists_df, text_col="playlist_name", prompt=playlist_prompt)
    playlist_df = playlist_df.head(20)
    print(playlist_df['playlist_name'].tolist())   

    # print to user that we are getting tracks
    message_placeholder.empty()
    message_placeholder.text("Getting tracks...")

    tracks_df = get_tracks(sp, playlist_df)
    tracks_df['text'] = tracks_df['categories'].apply(lambda x: ', '.join(x)) + ", " + tracks_df['artist_name']

    # sort tracks by similarity to prompt
    tracks_model = make_tracks_model(" ".join(tracks_df['categories'].explode().unique().tolist() + tracks_df['artist_name'].unique().tolist()))
    tracks_response = tracks_model.generate_content(prompt)
    tracks_prompt = tracks_response.text.strip()
    print(tracks_prompt)
    tracks_df = sort_by_cosine_similarity(tracks_df, text_col="text", prompt=tracks_prompt)
    tracks_df = tracks_df.head(50)
    st.session_state.tracks_df = tracks_df
    
    # print tracks to be added to the playlist
    message_placeholder.empty()

    st.write("### Tracks to be added to the playlist:")
    st.dataframe(tracks_df[['track_name', 'artist_name', 'track_date', 'track_popularity']])
    st.session_state.tracks_displayed = True

if st.session_state.tracks_displayed:
    if st.button('Create Playlist'):
        playlist_name = prompt
        playlist = sp.user_playlist_create(sp.me()['id'], playlist_name, public=False)
        # Add tracks to the playlist
        for uri in st.session_state.tracks_df['track_uri']:
            sp.playlist_add_items(playlist['id'], [uri])
        st.success(f"Playlist created: {playlist_name}")
        st.write(f"### [Open your playlist in Spotify]({playlist['external_urls']['spotify']})")
        st.session_state.tracks_displayed = False