import pandas as pd
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

def make_search_model():
    return genai.GenerativeModel(
        'gemini-1.5-flash',
        generation_config={"response_mime_type": "application/json"},
        system_instruction="""
        Spotify contains numerous genre/mood/decade/artist mixes curated for a given user. Your goal is to take in a user prompt asking for a playlist and return a list of Spotify search strings to find mixes that match the sentiment of the user prompt. The user prompt will be a string of text that describes the mood, theme, potential artists, and/or desired decade of the playlist they are looking for. You should then return a list of search strings that always end with "mix" and match the mood, theme, decade, artists of the user prompt. You should also designate whether the search string is for an artist mix or a mix relating to mood/genre/decade/etc. (we'll call the mixes relating to mood/genre/decade/etc. category mixes). 

        Your response should be in the following format:

        {
            "artist_strings": ["artist search string 1", "artist search string 2", "artist search string 3", ...],
            "category_strings": ["category search string 1", "category search string 2", "category search string 3", ...]
        }

        Guidelines:
        1. **Artist Strings**:
        - Only include artist strings if the user prompt explicitly mentions specific artists.
        - Format: "{artist} mix".
        - Exclude celebrities not primarily known for their music (e.g., actors, directors).
        - Be conservative with the number of artist strings.

        2. **Category Strings**:
        - Include strings that match the mood, theme, decade, or genre described in the prompt.
        - Format: "{category} mix".
        - Be liberal with the number of category strings, providing numerous options to cover the sentiment of the user prompt.

        3. **Be Liberal with Category Strings**:
        - Provide a generous number of category strings to match the sentiment of the user prompt.
        - Include various related moods, themes, decades, and genres.

        Examples:

        1. **User Prompt**: "I want to listen to upbeat 80s pop music."
        - Response:
            {
            "artist_strings": [],
            "category_strings": ["80s pop mix", "upbeat 80s mix", "80s dance mix", "80s party mix", "80s hits mix", "retro pop mix", "80s upbeat mix"]
            }

        2. **User Prompt**: "Songs that sound like I'm in a Quentin Tarantino movie."
        - Response:
            {
            "artist_strings": [],
            "category_strings": ["Tarantino movie soundtrack mix", "retro movie mix", "cinematic rock mix", "vintage vibes mix", "classic soundtrack mix", "70s rock mix", "eclectic mix", "movie soundtrack mix"]
            }

        3. **User Prompt**: "Play some relaxing jazz and blues from the 60s."
        - Response:
            {
            "artist_strings": [],
            "category_strings": ["60s jazz mix", "relaxing jazz mix", "60s blues mix", "jazz and blues mix", "smooth jazz mix", "vintage jazz mix", "60s chill mix", "classic blues mix"]
            }

        4. **User Prompt**: "I'm in the mood for some Taylor Swift songs."
        - Response:
            {
            "artist_strings": ["Taylor Swift mix"],
            "category_strings": ["pop mix", "country pop mix", "contemporary pop mix", "female pop mix", "pop hits mix", "pop anthems mix"]
            }

        Make sure to follow these guidelines strictly to ensure accurate and relevant search strings.
        """
    )

def make_playlist_model(categories_list):
    return genai.GenerativeModel(
        'gemini-1.5-flash',
        generation_config={"response_mime_type": "application/json"},
        system_instruction=f"""
        The user is trying to return a list of playlists that most closely match the sentiment of the user prompt based on the name of the playlist.
        The list of the playlist names is as follows:

        {", ".join(categories_list)}

        Your goal is to convert the user prompt into a phrase that is better for semantic search. The phrase should be a string of text that is more suitable for
        semantic search of the listed playlist names while still maintaining the sentiment of the user prompt. You should return a single string of text.
        """
    )

def make_tracks_model(categories_list):
    return genai.GenerativeModel(
        'gemini-1.5-flash',
        generation_config={"response_mime_type": "application/json"},
        system_instruction=f"""
        The user is trying to return a list of tracks that most closely match the sentiment of the user prompt based on the categories the track is in.
        The user has a text string for each track listing categories of the track. It is in the following format:

        category 1, category 2, category 3, ...

        The possible categories are as follows:

        {", ".join(categories_list)}

        Your goal is to convert the user prompt into a list of categories that can be used for semantic search. Your list should be a string of text that is more suitable for
        semantic search of the listed categories while still maintaining the sentiment of the user prompt.
        """
    )

def sort_by_cosine_similarity(data, text_col, prompt):
    response = genai.embed_content(
        model="models/embedding-001",
        content=[prompt] + data[text_col].tolist(),
        task_type="semantic_similarity"
    )
    embeddings = response['embedding']
    prompt_embedding = embeddings[0]
    data_embeddings = embeddings[1:]
    similarities = cosine_similarity([prompt_embedding], data_embeddings)[0]
    data['similarity'] = similarities
    return data.sort_values('similarity', ascending=False).reset_index(drop=True).copy()