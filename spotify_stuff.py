import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pandas as pd

def get_playlists(sp, search_strings):
    playlists = []
    for search_string in search_strings['category_strings']:
        result = sp.search(search_string, type='playlist', limit=50)
        for playlist in result['playlists']['items']:
            if playlist['owner']['id'] == 'spotify' and\
                (("/artist/" not in playlist['images'][0]['url']) and ("<a href=" not in playlist['description']) and not playlist['name'].endswith(" Radio")):
                playlists.append({
                    'playlist_id': playlist['id'],
                    'playlist_name': playlist['name'],
                    'playlist_description': playlist['description']
                })
    for search_string in search_strings['artist_strings']:
        result = sp.search(search_string, type='playlist', limit=50)
        for playlist in result['playlists']['items']:
            if playlist['owner']['id'] == 'spotify' and playlist['name'].lower().endswith(" mix"):
                playlists.append({
                    'playlist_id': playlist['id'],
                    'playlist_name': playlist['name'],
                    'playlist_description': playlist['description']
                })
    return playlists

def get_tracks(sp, data):
    tracks = []
    for idx, playlist in data.iterrows():
        result = sp.playlist_tracks(playlist['playlist_id'])
        for i, t in enumerate(result['items']):
            tracks.append({
                'categories': playlist['playlist_name'],
                'track_name': t['track']['name'],
                'artist_name': t['track']['artists'][0]['name'],
                'track_date': t['track']['album']['release_date'],
                'track_popularity': t['track']['popularity'],
                'track_uri': t['track']['uri']
            })
    tracks_df = pd.DataFrame(tracks)
    # Collapse the categories into a list
    tracks_df = tracks_df.groupby('track_uri').agg({
            'categories': list,
            'track_name': 'first',
            'artist_name': 'first',
            'track_date': 'first',
            'track_popularity': 'first'
        }).reset_index()
    return tracks_df