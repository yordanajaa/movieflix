import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup

# Paths
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
MOVIES_PATH = os.path.join(DATA_DIR, "movies.csv")
RATINGS_PATH = os.path.join(DATA_DIR, "ratings.csv")
LINKS_PATH = os.path.join(DATA_DIR, "links.csv")
SIMILARITY_CACHE_PATH = os.path.join(DATA_DIR, "item_similarity.pkl")
STATS_CACHE_PATH = os.path.join(DATA_DIR, "movie_stats.pkl")

def compute_movie_stats():
    """Computes average rating and rating count for all movies from ratings.csv in chunks."""
    if not os.path.exists(RATINGS_PATH):
        raise FileNotFoundError(f"ratings.csv not found at {RATINGS_PATH}")
    print("Computing movie statistics in chunks...")
    chunksize = 5_000_000
    movie_sums = {}
    movie_counts = {}
    for chunk in pd.read_csv(RATINGS_PATH, usecols=["movieId", "rating"], chunksize=chunksize):
        grouped = chunk.groupby("movieId")["rating"].agg(["sum", "count"])
        for movie_id, row in grouped.iterrows():
            movie_sums[movie_id] = movie_sums.get(movie_id, 0) + row["sum"]
            movie_counts[movie_id] = movie_counts.get(movie_id, 0) + row["count"]
            
    movie_ids = list(movie_counts.keys())
    averages = [movie_sums[mid] / movie_counts[mid] for mid in movie_ids]
    counts = [movie_counts[mid] for mid in movie_ids]
    
    stats_df = pd.DataFrame({
        "movieId": movie_ids,
        "avg_rating": averages,
        "num_ratings": counts
    })
    stats_df.to_pickle(STATS_CACHE_PATH)
    print(f"Movie statistics saved to {STATS_CACHE_PATH}")
    return stats_df

def load_movies():
    """Loads movies.csv, calculates/loads rating stats, merges links, and returns a DataFrame."""
    if not os.path.exists(MOVIES_PATH):
        raise FileNotFoundError(f"movies.csv not found at {MOVIES_PATH}")
    
    df = pd.read_csv(MOVIES_PATH)
    # Fill NaN genres with empty string
    df['genres'] = df['genres'].fillna('')
    # Replace pipe with space for TF-IDF representation
    df['genres_processed'] = df['genres'].str.replace('|', ' ', regex=False)
    # Extract release year from title e.g. "Toy Story (1995)" -> 1995
    df['year'] = df['title'].str.extract(r'\((\d{4})\)$')
    # If no year is found, set to unknown or 0
    df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)
    
    # Load or compute statistics
    if os.path.exists(STATS_CACHE_PATH):
        stats_df = pd.read_pickle(STATS_CACHE_PATH)
    else:
        stats_df = compute_movie_stats()
        
    # Merge stats into movies DataFrame
    df = df.merge(stats_df, on="movieId", how="left")
    df['avg_rating'] = df['avg_rating'].fillna(0.0)
    df['num_ratings'] = df['num_ratings'].fillna(0).astype(int)
    
    # Merge imdbId and tmdbId from links.csv
    if os.path.exists(LINKS_PATH):
        try:
            links_df = pd.read_csv(LINKS_PATH, usecols=["movieId", "imdbId", "tmdbId"])
            links_df['tmdbId'] = links_df['tmdbId'].astype('Int64')
            links_df['imdbId'] = links_df['imdbId'].astype('Int64')
            df = df.merge(links_df, on="movieId", how="left")
        except Exception as e:
            print(f"Error loading links.csv: {e}")
            df['tmdbId'] = pd.Series(dtype='Int64')
            df['imdbId'] = pd.Series(dtype='Int64')
    else:
        df['tmdbId'] = pd.Series(dtype='Int64')
        df['imdbId'] = pd.Series(dtype='Int64')
        
    return df

def fetch_tmdb_details(tmdb_id):
    """Scrapes movie poster URL and synopsis/description from TMDb by tmdbId.
    Returns: (poster_url, description)
    """
    if not tmdb_id or pd.isna(tmdb_id):
        return None, "Deskripsi film tidak tersedia."
        
    try:
        tmdb_id_int = int(tmdb_id)
        url = f"https://www.themoviedb.org/movie/{tmdb_id_int}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code != 200:
            return None, "Gagal memuat deskripsi dari TMDb."
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract description
        desc_div = soup.find("div", class_="overview")
        description = desc_div.get_text().strip() if desc_div else "Deskripsi cerita belum tersedia untuk film ini."
        
        # Extract poster URL
        poster_img = soup.find("img", class_="poster")
        poster_url = None
        if poster_img:
            poster_url = poster_img.get('src') or poster_img.get('data-src')
        else:
            # Fallback search in img tags
            for img in soup.find_all("img"):
                src = img.get('src', '')
                if 'poster' in src or 't/p/w' in src:
                    poster_url = src
                    break
                    
        # Make absolute if relative
        if poster_url and poster_url.startswith('/'):
            poster_url = "https://image.tmdb.org" + poster_url
        elif poster_url and poster_url.startswith('//'):
            poster_url = "https:" + poster_url
            
        return poster_url, description
    except Exception as e:
        return None, f"Gagal memuat informasi film ({e})."

def get_content_model(movies_df):
    """Fits TF-IDF vectorizer on movie genres."""
    vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b') # keep single letters like 'Sci-Fi'
    tfidf_matrix = vectorizer.fit_transform(movies_df['genres_processed'])
    return tfidf_matrix

def recommend_by_content(movie_id, movies_df, tfidf_matrix, top_n=10):
    """Recommends movies similar to movie_id based on genres."""
    if movie_id not in movies_df['movieId'].values:
        return pd.DataFrame()
        
    # Get idx of the movie
    idx = movies_df[movies_df['movieId'] == movie_id].index[0]
    
    # Calculate similarity on-the-fly for this movie to save memory
    movie_vector = tfidf_matrix[idx]
    similarities = cosine_similarity(movie_vector, tfidf_matrix).flatten()
    
    # Get top_n indices (excluding the movie itself)
    sim_indices = similarities.argsort()[::-1]
    sim_indices = [i for i in sim_indices if i != idx]
    
    # Get the recommendations
    recs = movies_df.iloc[sim_indices].copy()
    recs['similarity'] = similarities[sim_indices]
    
    return recs.head(top_n)[['movieId', 'title', 'genres', 'year', 'similarity', 'avg_rating', 'num_ratings', 'tmdbId', 'imdbId']]

def build_collaborative_model(min_movie_ratings=4000, min_user_ratings=300):
    """Reads ratings.csv, filters it, computes item similarity, and saves it."""
    if not os.path.exists(RATINGS_PATH):
        raise FileNotFoundError(f"ratings.csv not found at {RATINGS_PATH}")
    
    print("Analyzing rating counts to filter dataset...")
    # Load in chunks only to count to be memory safe
    chunksize = 5_000_000
    movie_counts = pd.Series(dtype=int)
    user_counts = pd.Series(dtype=int)
    
    for chunk in pd.read_csv(RATINGS_PATH, usecols=["userId", "movieId"], chunksize=chunksize):
        movie_counts = movie_counts.add(chunk['movieId'].value_counts(), fill_value=0)
        user_counts = user_counts.add(chunk['userId'].value_counts(), fill_value=0)
        
    popular_movies = movie_counts[movie_counts >= min_movie_ratings].index
    active_users = user_counts[user_counts >= min_user_ratings].index
    
    print(f"Kept {len(popular_movies)} popular movies and {len(active_users)} active users.")
    
    # Filter and load ratings
    print("Loading and filtering ratings.csv...")
    filtered_chunks = []
    for chunk in pd.read_csv(RATINGS_PATH, usecols=["userId", "movieId", "rating"], chunksize=chunksize):
        filtered_chunk = chunk[chunk['movieId'].isin(popular_movies) & chunk['userId'].isin(active_users)]
        filtered_chunks.append(filtered_chunk)
        
    df_filtered = pd.concat(filtered_chunks, ignore_index=True)
    
    print("Creating utility matrix (pivot table)...")
    pivot_table = df_filtered.pivot(index='userId', columns='movieId', values='rating')
    pivot_norm = pivot_table.fillna(0)
    
    print("Computing item-to-item similarity...")
    item_sim = cosine_similarity(pivot_norm.T)
    item_sim_df = pd.DataFrame(item_sim, index=pivot_table.columns, columns=pivot_table.columns)
    
    # Save cache
    item_sim_df.to_pickle(SIMILARITY_CACHE_PATH)
    print(f"Similarity matrix saved to {SIMILARITY_CACHE_PATH}")
    return item_sim_df

def load_collaborative_model():
    """Loads similarity matrix from cache or builds it if cache doesn't exist."""
    if os.path.exists(SIMILARITY_CACHE_PATH):
        print(f"Loading item similarity from cache {SIMILARITY_CACHE_PATH}...")
        try:
            return pd.read_pickle(SIMILARITY_CACHE_PATH)
        except Exception as e:
            print(f"Failed to read cache: {e}. Rebuilding...")
    return build_collaborative_model()

def recommend_by_collaborative(movie_id, item_sim_df, movies_df, top_n=10):
    """Recommends movies similar to movie_id based on collaborative filtering."""
    if movie_id not in item_sim_df.index:
        return pd.DataFrame() # Not in popular list
        
    # Get similarity vector for this movie
    sim_scores = item_sim_df[movie_id].drop(index=movie_id, errors='ignore').sort_values(ascending=False)
    
    # Get top_n similarities
    sim_scores_top = sim_scores.head(top_n)
    
    # Join with movies details
    recs = pd.DataFrame({'similarity': sim_scores_top})
    recs = recs.join(movies_df.set_index('movieId'), how='inner')
    recs = recs.reset_index().rename(columns={'index': 'movieId'})
    
    # Sort by similarity again (since join might change order)
    recs = recs.sort_values(by='similarity', ascending=False)
    
    return recs[['movieId', 'title', 'genres', 'year', 'similarity', 'avg_rating', 'num_ratings', 'tmdbId', 'imdbId']]
