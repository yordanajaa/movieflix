import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import recommender
import recommender_metrics
from concurrent.futures import ThreadPoolExecutor
import warnings
import logging

# Suppress WebSocket warnings
logging.getLogger('tornado').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# Parse URL query parameters
query_params = st.query_params
selected_movie_id_from_url = None
if "movieId" in query_params:
    try:
        selected_movie_id_from_url = int(query_params["movieId"])
    except (ValueError, TypeError):
        pass

# Set page configuration with a premium Netflix-like title and icon
st.set_page_config(
    page_title="Movieflix AI - Sistem Rekomendasi Film",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Netflix-Themed Premium CSS Styling with Fresh Animations
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Plus+Jakarta+Sans:wght@300;400;600;700&display=swap');
    
    /* ========== ANIMATIONS ========== */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes glowPulse {
        0%, 100% {
            box-shadow: 0 0 20px rgba(229, 9, 20, 0.4), 0 8px 40px rgba(0, 0, 0, 0.5);
        }
        50% {
            box-shadow: 0 0 40px rgba(229, 9, 20, 0.6), 0 12px 50px rgba(0, 0, 0, 0.7);
        }
    }
    
    @keyframes shimmer {
        0% {
            background-position: -1000px 0;
        }
        100% {
            background-position: 1000px 0;
        }
    }
    
    @keyframes gradientShift {
        0% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
        100% {
            background-position: 0% 50%;
        }
    }
    
    @keyframes scaleHover {
        0% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
        100% {
            transform: scale(1);
        }
    }
    
    /* ========== MAIN STYLING ========== */
    /* Main body overrides */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 50%, #0f0f0f 100%);
        color: #e5e5e5 !important;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f0f0f 0%, #1a1a1a 50%, #0f0f0f 100%) !important;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        color: #ffffff !important;
        font-weight: 800;
        animation: fadeInUp 0.8s ease-out;
    }
    
    /* ========== BRANDING ========== */
    .brand-logo {
        background: linear-gradient(135deg, #E50914 0%, #ff4757 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-family: 'Outfit', sans-serif;
        font-size: 3rem;
        font-weight: 900;
        letter-spacing: -2px;
        text-transform: uppercase;
        margin-bottom: 0.1rem;
        display: inline-block;
        animation: slideInLeft 0.8s cubic-bezier(0.34, 1.56, 0.64, 1);
    }
    
    .brand-sub {
        font-size: 1rem;
        background: linear-gradient(90deg, #aaa 0%, #E50914 50%, #aaa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 2rem;
        font-weight: 400;
        animation: fadeInUp 1s ease-out 0.2s both;
    }
    
    /* ========== HERO BANNER ========== */
    .hero-banner {
        background: linear-gradient(135deg, rgba(30,30,30,0.98) 0%, rgba(20,20,20,0.95) 50%, rgba(30,30,30,0.98) 100%);
        border: 2px solid;
        border-image: linear-gradient(135deg, #E50914 0%, #ff4757 100%) 1;
        border-radius: 16px;
        padding: 2.5rem;
        margin-bottom: 2.5rem;
        box-shadow: 0 8px 40px rgba(229, 9, 20, 0.15), 0 0 60px rgba(229, 9, 20, 0.05);
        animation: fadeInUp 0.8s ease-out 0.3s both;
        backdrop-filter: blur(10px);
        position: relative;
        overflow: hidden;
    }
    
    .hero-banner::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(229, 9, 20, 0.05) 0%, transparent 100%);
        pointer-events: none;
        border-radius: 16px;
    }
    
    .hero-poster-container {
        border-radius: 12px;
        box-shadow: 0 20px 60px rgba(229, 9, 20, 0.25), 0 0 40px rgba(229, 9, 20, 0.1);
        overflow: hidden;
        border: 2px solid rgba(229, 9, 20, 0.3);
        animation: glowPulse 3s ease-in-out infinite;
        transition: transform 0.3s ease;
    }
    
    .hero-poster-container:hover {
        transform: scale(1.02);
    }
    
    .hero-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #ffffff 0%, #E50914 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        line-height: 1.1;
        margin-bottom: 0.5rem;
    }
    
    .hero-meta {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1.5rem;
        flex-wrap: wrap;
    }
    
    .hero-year {
        font-size: 1.1rem;
        color: #E50914;
        font-weight: bold;
        padding: 0.4rem 0.8rem;
        background: rgba(229, 9, 20, 0.1);
        border-radius: 6px;
        border: 1px solid rgba(229, 9, 20, 0.3);
    }
    
    .hero-match {
        font-size: 1rem;
        color: #4ade80;
        font-weight: 700;
        padding: 0.4rem 0.8rem;
        background: rgba(74, 222, 128, 0.1);
        border-radius: 6px;
        border: 1px solid rgba(74, 222, 128, 0.3);
    }
    
    .hero-cf-badge {
        font-size: 0.9rem;
        font-weight: 700;
        color: #fbbf24;
        background: rgba(251, 191, 36, 0.15);
        padding: 0.4rem 0.8rem;
        border-radius: 6px;
        border: 1px solid rgba(251, 191, 36, 0.3);
    }
    
    .hero-desc {
        font-size: 1.1rem;
        color: #d1d5db;
        line-height: 1.8;
        margin-bottom: 1.5rem;
    }
    
    /* ========== MOVIE CARDS ========== */
    .movie-card {
        background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
        border: 1px solid rgba(229, 9, 20, 0.2);
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.6), 0 0 40px rgba(229, 9, 20, 0.05);
        transition: all 0.4s cubic-bezier(0.165, 0.84, 0.44, 1);
        margin-bottom: 1.5rem;
        height: 100%;
        display: flex;
        flex-direction: column;
        position: relative;
        animation: fadeInUp 0.6s ease-out backwards;
    }
    
    .movie-card:nth-child(1) { animation-delay: 0s; }
    .movie-card:nth-child(2) { animation-delay: 0.1s; }
    .movie-card:nth-child(3) { animation-delay: 0.2s; }
    .movie-card:nth-child(4) { animation-delay: 0.3s; }
    .movie-card:nth-child(5) { animation-delay: 0.4s; }
    
    .movie-card:hover {
        transform: translateY(-12px) scale(1.05);
        border-color: #E50914;
        box-shadow: 0 20px 50px rgba(229, 9, 20, 0.35), 0 0 60px rgba(229, 9, 20, 0.15);
    }
    
    .movie-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(229, 9, 20, 0.1) 0%, transparent 100%);
        opacity: 0;
        transition: opacity 0.3s ease;
        pointer-events: none;
        border-radius: 12px;
    }
    
    .movie-card:hover::before {
        opacity: 1;
    }
    
    .movie-card-img-container {
        position: relative;
        width: 100%;
        aspect-ratio: 2/3;
        overflow: hidden;
        background-color: #2a2a2a;
        background-image: linear-gradient(90deg, #3a3a3a 25%, #4a4a4a 50%, #3a3a3a 75%);
        background-size: 200% 100%;
        animation: shimmer 2s infinite;
    }
    
    .movie-card-img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        transition: transform 0.4s ease;
    }
    
    .movie-card:hover .movie-card-img {
        transform: scale(1.1);
    }
    
    .movie-card-content {
        padding: 1.2rem;
        display: flex;
        flex-direction: column;
        flex-grow: 1;
        position: relative;
        z-index: 1;
    }
    
    .card-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 0.5rem;
        line-height: 1.3;
        height: 3.0rem;
        display: -webkit-box;
        -webkit-line-clamp: 2;
        -webkit-box-orient: vertical;
        overflow: hidden;
        transition: color 0.3s ease;
    }
    
    .movie-card:hover .card-title {
        color: #E50914;
    }
    
    .card-meta {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 0.8rem;
    }
    
    .card-match {
        font-size: 0.9rem;
        color: #4ade80;
        font-weight: 700;
    }
    
    .card-year {
        font-size: 0.85rem;
        color: #E50914;
        background: rgba(229, 9, 20, 0.15);
        padding: 0.3rem 0.6rem;
        border-radius: 6px;
        border: 1px solid rgba(229, 9, 20, 0.3);
        font-weight: 600;
    }
    
    .card-rating-row {
        font-size: 0.9rem;
        color: #fbbf24;
        margin-bottom: 0.8rem;
        font-weight: 600;
    }
    
    .card-desc {
        font-size: 0.85rem;
        color: #b0b9c6;
        line-height: 1.5;
        margin-bottom: 1rem;
        height: 3.6rem;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
        overflow: hidden;
    }
    
    .card-genres {
        display: flex;
        flex-wrap: wrap;
        gap: 0.4rem;
        margin-top: auto;
    }
    
    /* ========== GENRE BADGES ========== */
    .genre-badge {
        font-size: 0.75rem;
        font-weight: 600;
        color: #E50914;
        background: rgba(229, 9, 20, 0.15);
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        border: 1px solid rgba(229, 9, 20, 0.3);
        transition: all 0.3s ease;
    }
    
    .genre-badge:hover {
        background: rgba(229, 9, 20, 0.25);
        border-color: #E50914;
        transform: scale(1.05);
    }
    
    /* ========== SIDEBAR ========== */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #181818 0%, #0f0f0f 100%) !important;
        border-right: 1px solid rgba(229, 9, 20, 0.2);
    }
    
    /* ========== TABS ========== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        border-bottom: 2px solid rgba(229, 9, 20, 0.2);
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Outfit', sans-serif;
        font-size: 1.1rem;
        color: #888888;
        background-color: transparent !important;
        padding-bottom: 0.75rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #E50914;
    }
    
    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
        border-bottom-color: #E50914 !important;
        font-weight: bold;
    }
    
    /* ========== BUTTONS ========== */
    .stButton>button {
        background: linear-gradient(135deg, #E50914 0%, #ff4757 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: bold !important;
        padding: 0.7rem 1.8rem !important;
        transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
        font-family: 'Outfit', sans-serif !important;
        box-shadow: 0 4px 15px rgba(229, 9, 20, 0.3) !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(229, 9, 20, 0.5) !important;
    }
    
    .stButton>button:active {
        transform: translateY(0) !important;
    }
    
    /* ========== METRIC BOX ========== */
    .metric-box {
        background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
        border: 1px solid rgba(229, 9, 20, 0.2);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        height: 140px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        box-sizing: border-box;
        transition: all 0.4s ease;
        animation: fadeInUp 0.6s ease-out;
    }
    
    .metric-box:hover {
        transform: translateY(-8px);
        border-color: #E50914;
        box-shadow: 0 12px 30px rgba(229, 9, 20, 0.25);
    }
    
    /* ========== LINKS ========== */
    a[target="_self"] {
        text-decoration: none !important;
        color: inherit !important;
        display: block;
        height: 100%;
    }
    
    a[target="_self"]:hover {
        text-decoration: none !important;
        color: inherit !important;
    }
    
    /* ========== LOADING ANIMATION ========== */
    .spinner {
        border: 4px solid rgba(229, 9, 20, 0.2);
        border-top: 4px solid #E50914;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* ========== RESPONSIVE ========== */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2rem;
        }
        
        .brand-logo {
            font-size: 2rem;
        }
        
        .hero-meta {
            flex-direction: column;
            align-items: flex-start;
        }
    }
</style>
""", unsafe_allow_html=True)

# ----------------- DATA LOADING & CACHING -----------------

@st.cache_data
def get_cached_movies_v2():
    return recommender.load_movies()

@st.cache_resource
def get_cached_content_matrix(movies_df):
    return recommender.get_content_model(movies_df)

@st.cache_resource
def get_cached_collaborative_matrix():
    return recommender.load_collaborative_model()

# Loading details helper with caching (BeautifulSoup)
@st.cache_data(show_spinner=False)
def get_movie_details_cached(tmdb_id):
    """Cached details retrieval (poster url and plot overview)."""
    poster, desc = recommender.fetch_tmdb_details(tmdb_id)
    if not poster:
        # High quality aesthetic cinematic poster placeholder
        poster = "https://images.unsplash.com/photo-1594909122845-11baa439b7bf?q=80&w=300&auto=format&fit=crop"
    return poster, desc

def load_details_in_parallel(recs_df):
    """Loads TMDb details (posters, synopses) concurrently to ensure maximum performance."""
    if recs_df.empty:
        return []
    records = list(recs_df.itertuples())
    with ThreadPoolExecutor(max_workers=len(records)) as executor:
        results = list(executor.map(lambda r: get_movie_details_cached(r.tmdbId), records))
    return results

# Load Core Data
with st.spinner("🚀 Memuat database film & konfigurasi sistem..."):
    try:
        movies_df = get_cached_movies_v2()
        tfidf_matrix = get_cached_content_matrix(movies_df)
        item_sim_df = get_cached_collaborative_matrix()
    except Exception as e:
        st.error(f"Gagal memuat dataset: {e}")
        st.stop()

# ----------------- SIDEBAR PANEL -----------------

st.sidebar.markdown("<div style='color: #E50914; font-family: Outfit; font-size: 2rem; font-weight: 900; letter-spacing:-1px; text-transform:uppercase; margin-bottom: 1rem;'>MOVIEFLIX</div>", unsafe_allow_html=True)
st.sidebar.subheader("Pilih Film Favorit Anda")

# Get unique genres list
all_genres = movies_df['genres'].str.split('|').explode()
unique_genres = sorted(all_genres[(all_genres != '(no genres listed)') & (all_genres.notna())].unique().tolist())

selected_genre = st.sidebar.selectbox(
    "Filter berdasarkan Genre:",
    options=["Semua Genre"] + unique_genres
)

# Filter movies based on selected genre
if selected_genre != "Semua Genre":
    filtered_df = movies_df[movies_df['genres'].str.contains(selected_genre, case=False, na=False)]
else:
    filtered_df = movies_df

# Setup selectbox for movie selection based on filtered list
movie_titles = filtered_df['title'].tolist()
movie_to_id = dict(zip(filtered_df['title'], filtered_df['movieId']))
id_to_title = dict(zip(filtered_df['movieId'], filtered_df['title']))

# Determine the selectbox index based on URL movie ID
default_index = 0
selectbox_key = f"movie_select_{selected_genre}"

if selected_movie_id_from_url is not None:
    if selected_movie_id_from_url in id_to_title:
        target_title = id_to_title[selected_movie_id_from_url]
        if target_title in movie_titles:
            default_index = movie_titles.index(target_title) + 1
    else:
        if selectbox_key in st.session_state:
            del st.session_state[selectbox_key]

if len(movie_titles) > 0:
    options_list = ["Semua"] + movie_titles
    selected_title = st.sidebar.selectbox(
        "Cari judul film:",
        options=options_list,
        index=default_index,
        key=selectbox_key
    )
    if selected_title == "Semua":
        # If the selectbox is changed to "Semua" manually, clear URL params
        if selected_movie_id_from_url is not None and selected_movie_id_from_url in id_to_title:
            st.query_params.clear()
            st.rerun()
        # If there is a selected movie from URL but it's not in this genre filter, we still view it
        if selected_movie_id_from_url is not None:
            is_browse_mode = False
            selected_movie_id = selected_movie_id_from_url
            selected_movie_info = movies_df[movies_df['movieId'] == selected_movie_id].iloc[0]
        else:
            is_browse_mode = True
    else:
        is_browse_mode = False
        selected_movie_id = movie_to_id[selected_title]
        selected_movie_info = movies_df[movies_df['movieId'] == selected_movie_id].iloc[0]
        # Sync URL if sidebar selection changed
        if selected_movie_id_from_url != selected_movie_id:
            st.query_params["movieId"] = str(selected_movie_id)
            st.rerun()
else:
    st.sidebar.warning("Tidak ada film di genre ini.")
    st.stop()

# Tampilkan semua rekomendasi yang relevan (dibatasi 24 film demi performa pemuatan gambar)
top_n = 24

st.sidebar.markdown("---")

# ----------------- MAIN PAGES & HERO BANNER -----------------

st.markdown("<div class='brand-logo'>Movieflix</div>", unsafe_allow_html=True)
st.markdown("<div class='brand-sub'>Sistem rekomendasi film cerdas dengan citra poster dan sinopsis sinematik</div>", unsafe_allow_html=True)

if not is_browse_mode:
    # Fetch Selected Movie Poster & Description
    with st.spinner("Mengunduh poster film terpilih..."):
        selected_poster, selected_desc = get_movie_details_cached(selected_movie_info['tmdbId'])

    # Render Cinema Hero Banner (Netflix Style)
    is_in_cf = selected_movie_id in item_sim_df.index
    cf_status_text = "POPULER DI USER" if is_in_cf else "KLASIK/INDIE"
    selected_stars = "".join(["★" for _ in range(int(round(selected_movie_info['avg_rating'])))]) + "".join(["☆" for _ in range(5 - int(round(selected_movie_info['avg_rating'])))])

    # Add a beautiful back button if viewing detail via query parameters
    col_back, _ = st.columns([1, 4])
    with col_back:
        if st.button("← Kembali ke Galeri", key="back_to_gallery", use_container_width=True):
            st.query_params.clear()
            for k in list(st.session_state.keys()):
                if k.startswith("movie_select_"):
                    del st.session_state[k]
            st.rerun()

    st.markdown("### 🎬 Detail Film")
    col_hero_p, col_hero_d = st.columns([1, 4])
    with col_hero_p:
        st.markdown(f"""
        <div class='hero-poster-container'>
            <img src='{selected_poster}' style='width: 100%; aspect-ratio: 2/3; object-fit: cover; display: block;'>
        </div>
        """, unsafe_allow_html=True)

    with col_hero_d:
        selected_genres = selected_movie_info['genres'].split('|')
        genre_badges = "".join([f"<span class='genre-badge'>{g}</span>" for g in selected_genres])
        
        # Build watch and detail links safely to avoid KeyError on stale caches
        clean_title = selected_movie_info['title'].split(' (')[0]
        google_watch_url = f"https://www.google.com/search?q=Nonton+{clean_title.replace(' ', '+')}+sub+Indo"
        youtube_trailer_url = f"https://www.youtube.com/results?search_query={clean_title.replace(' ', '+')}+official+trailer"
        
        tmdb_id_val = selected_movie_info.get('tmdbId') if 'tmdbId' in selected_movie_info else None
        tmdb_url = f"https://www.themoviedb.org/movie/{int(tmdb_id_val)}" if tmdb_id_val is not None and not pd.isna(tmdb_id_val) else None
        
        imdb_id_val = selected_movie_info.get('imdbId') if 'imdbId' in selected_movie_info else None
        imdb_url = f"https://www.imdb.com/title/tt{int(imdb_id_val):07d}/" if imdb_id_val is not None and not pd.isna(imdb_id_val) else None
        
        buttons_html = f"<div style='display: flex; gap: 0.8rem; flex-wrap: wrap; margin-top: 1.5rem;'><a href='{google_watch_url}' target='_blank' style='text-decoration: none;'><button style='background-color: #E50914; color: white; border: none; padding: 0.6rem 1.2rem; font-size: 0.95rem; font-weight: bold; border-radius: 4px; cursor: pointer; display: flex; align-items: center; gap: 6px; transition: background-color 0.2s;'>🔴 Nonton Sekarang (Google)</button></a><a href='{youtube_trailer_url}' target='_blank' style='text-decoration: none;'><button style='background-color: #2f2f2f; color: white; border: 1px solid #444; padding: 0.6rem 1.2rem; font-size: 0.95rem; font-weight: bold; border-radius: 4px; cursor: pointer; display: flex; align-items: center; gap: 6px; transition: background-color 0.2s;'>📺 Trailer (YouTube)</button></a>"
        
        if tmdb_url:
            buttons_html += f"<a href='{tmdb_url}' target='_blank' style='text-decoration: none;'><button style='background-color: #0d253f; color: #01b4e4; border: 1px solid #01b4e4; padding: 0.6rem 1.2rem; font-size: 0.95rem; font-weight: bold; border-radius: 4px; cursor: pointer; display: flex; align-items: center; gap: 6px; transition: background-color 0.2s;'>🎬 Halaman TMDb</button></a>"
            
        if imdb_url:
            buttons_html += f"<a href='{imdb_url}' target='_blank' style='text-decoration: none;'><button style='background-color: #f5c518; color: black; border: none; padding: 0.6rem 1.2rem; font-size: 0.95rem; font-weight: bold; border-radius: 4px; cursor: pointer; display: flex; align-items: center; gap: 6px; transition: background-color 0.2s;'>ℹ️ Halaman IMDb</button></a>"
            
        buttons_html += "</div>"
        
        st.markdown(f"""
        <div class='hero-banner'>
            <div class='hero-title'>{selected_movie_info['title']}</div>
            <div class='hero-meta'>
                <span class='hero-year'>{selected_movie_info['year'] if selected_movie_info['year'] > 0 else 'N/A'}</span>
                <span style='color: #fbbf24; font-weight: bold;'>{selected_stars} {selected_movie_info['avg_rating']:.2f}/5.0</span>
                <span style='color: #888;'>({int(selected_movie_info['num_ratings']):,} voting)</span>
                <span class='hero-match'>98% Cocok</span>
                <span class='hero-cf-badge'>{cf_status_text}</span>
            </div>
            <div class='hero-desc'>{selected_desc}</div>
            <div style='margin-top: 1rem;'>
                <span style='color: #999; font-size: 0.9rem; font-weight: bold; margin-right: 0.5rem;'>GENRE:</span>
                {genre_badges}
            </div>
            {buttons_html}
        </div>
        """, unsafe_allow_html=True)
else:
    st.markdown(f"### 🎬 Kategori Film: {selected_genre}")

# Tabs Navigation
tab1, tab2, tab3 = st.tabs([
    "🎯 Rekomendasi Genre Serupa", 
    "👥 Rekomendasi Pilihan Penonton", 
    "📊 Analitik Dataset"
])

# ----------------- TAB 1: CONTENT-BASED FILTERING -----------------
with tab1:
    if is_browse_mode:
        st.markdown(f"#### Jelajahi Semua Film Terpopuler - {selected_genre}")
        st.write(f"Daftar film paling populer di genre **{selected_genre}** berdasarkan jumlah rating terbanyak dari penonton.")
        
        # Get top 24 movies in this genre sorted by num_ratings descending
        browse_movies = filtered_df.sort_values(by='num_ratings', ascending=False).head(top_n)
        
        if browse_movies.empty:
            st.info("Tidak ada film yang ditemukan untuk kategori ini.")
        else:
            with st.spinner("Mengunduh poster film..."):
                browse_details = load_details_in_parallel(browse_movies)
                
            cols = st.columns(4)
            for i, row in enumerate(browse_movies.itertuples()):
                col_target = cols[i % 4]
                poster_url, description = browse_details[i]
                
                item_stars = "".join(["★" for _ in range(int(round(row.avg_rating)))]) + "".join(["☆" for _ in range(5 - int(round(row.avg_rating)))])
                item_genres = row.genres.split('|')[:3]
                genres_html = "".join([f"<span class='genre-badge'>{g}</span>" for g in item_genres])
                
                with col_target:
                    st.markdown(f"""
                    <a href='/?movieId={row.movieId}' target='_self' style='text-decoration: none; color: inherit;'>
                        <div class='movie-card'>
                            <div class='movie-card-img-container'>
                                <img class='movie-card-img' src='{poster_url}'>
                            </div>
                            <div class='movie-card-content'>
                                <div class='card-title'>{row.title}</div>
                                <div class='card-meta'>
                                    <span style='color: #E50914; font-weight: bold; font-size: 0.85rem;'>TRENDING</span>
                                    <span class='card-year'>{row.year if row.year > 0 else 'N/A'}</span>
                                </div>
                                <div class='card-rating-row'>
                                    {item_stars} <span style='color: #cbd5e1; font-weight: bold;'>{row.avg_rating:.1f}</span>
                                    <span style='color: #777; font-size: 0.75rem;'>({int(row.num_ratings):,})</span>
                                </div>
                                <div class='card-desc'>{description}</div>
                                <div class='card-genres'>{genres_html}</div>
                            </div>
                        </div>
                    </a>
                    """, unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='margin-bottom: 0.2rem;'>Karena Anda Menonton " + selected_movie_info['title'] + "</h3>", unsafe_allow_html=True)
        st.write("Rekomendasi genre sejenis berdasarkan kemiripan representasi teks TF-IDF.")
        
        with st.spinner("Menghitung rekomendasi genre..."):
            recs_content = recommender.recommend_by_content(selected_movie_id, movies_df, tfidf_matrix, top_n)
            
        if recs_content.empty:
            st.info("Tidak ada rekomendasi content-based yang ditemukan.")
        else:
            # Tampilkan metrik presisi di UI
            cb_metrics = recommender_metrics.get_recommendation_metrics(recs_content, movies_df, "Content-Based")
            st.markdown(f"""
            <div style='background: rgba(229, 9, 20, 0.1); border-left: 4px solid #E50914; padding: 1rem; border-radius: 4px; margin-bottom: 1.5rem;'>
                <span style='color: #fff; font-weight: bold; margin-right: 15px;'>🧪 Uji Presisi Model:</span>
                <span style='color: #4ade80; font-weight: bold;'>Precision@10: {cb_metrics['precision@10']*100:.1f}%</span> | 
                <span style='color: #fbbf24; font-weight: bold;'>Diversity Score: {cb_metrics['diversity_score']:.2f}</span> | 
                <span style='color: #60a5fa; font-weight: bold;'>Avg Quality Rating: {cb_metrics['avg_rating']:.1f}⭐</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Fetch posters/descriptions concurrently
            with st.spinner("Mengunduh poster film rekomendasi..."):
                recs_details = load_details_in_parallel(recs_content)
                
            # Display as a Netflix Grid (4 columns)
            cols = st.columns(4)
            for i, row in enumerate(recs_content.itertuples()):
                col_target = cols[i % 4]
                poster_url, description = recs_details[i]
                
                # Format rating stars
                item_stars = "".join(["★" for _ in range(int(round(row.avg_rating)))]) + "".join(["☆" for _ in range(5 - int(round(row.avg_rating)))])
                match_pct = int(row.similarity * 100)
                
                item_genres = row.genres.split('|')[:3] # Show max 3 genres on card
                genres_html = "".join([f"<span class='genre-badge'>{g}</span>" for g in item_genres])
                
                with col_target:
                    st.markdown(f"""
                    <a href='/?movieId={row.movieId}' target='_self' style='text-decoration: none; color: inherit;'>
                        <div class='movie-card'>
                            <div class='movie-card-img-container'>
                                <img class='movie-card-img' src='{poster_url}'>
                            </div>
                            <div class='movie-card-content'>
                                <div class='card-title'>{row.title}</div>
                                <div class='card-meta'>
                                    <span class='card-match'>{match_pct}% Match</span>
                                    <span class='card-year'>{row.year if row.year > 0 else 'N/A'}</span>
                                </div>
                                <div class='card-rating-row'>
                                    {item_stars} <span style='color: #cbd5e1; font-weight: bold;'>{row.avg_rating:.1f}</span>
                                    <span style='color: #777; font-size: 0.75rem;'>({int(row.num_ratings):,})</span>
                                </div>
                                <div class='card-desc'>{description}</div>
                                <div class='card-genres'>{genres_html}</div>
                            </div>
                        </div>
                    </a>
                    """, unsafe_allow_html=True)

# ----------------- TAB 2: COLLABORATIVE FILTERING -----------------
with tab2:
    if is_browse_mode:
        st.markdown("#### Film Terpopuler Sepanjang Masa (Semua Kategori)")
        st.write("Saran film paling populer dan memiliki ulasan terbanyak dari seluruh kategori/genre di database MovieLens.")
        
        # Get overall top 24 movies sorted by num_ratings descending
        browse_overall = movies_df.sort_values(by='num_ratings', ascending=False).head(top_n)
        
        if browse_overall.empty:
            st.info("Tidak ada film terpopuler yang ditemukan.")
        else:
            with st.spinner("Mengunduh poster film terpopuler..."):
                browse_details_overall = load_details_in_parallel(browse_overall)
                
            cols = st.columns(4)
            for i, row in enumerate(browse_overall.itertuples()):
                col_target = cols[i % 4]
                poster_url, description = browse_details_overall[i]
                
                item_stars = "".join(["★" for _ in range(int(round(row.avg_rating)))]) + "".join(["☆" for _ in range(5 - int(round(row.avg_rating)))])
                
                item_genres = row.genres.split('|')[:3]
                genres_html = "".join([f"<span class='genre-badge'>{g}</span>" for g in item_genres])
                
                with col_target:
                    st.markdown(f"""
                    <a href='/?movieId={row.movieId}' target='_self' style='text-decoration: none; color: inherit;'>
                        <div class='movie-card'>
                            <div class='movie-card-img-container'>
                                <img class='movie-card-img' src='{poster_url}'>
                            </div>
                            <div class='movie-card-content'>
                                <div class='card-title'>{row.title}</div>
                                <div class='card-meta'>
                                    <span style='color: #46d369; font-weight: bold; font-size: 0.85rem;'>TRENDING GLOBAL</span>
                                    <span class='card-year'>{row.year if row.year > 0 else 'N/A'}</span>
                                </div>
                                <div class='card-rating-row'>
                                    {item_stars} <span style='color: #cbd5e1; font-weight: bold;'>{row.avg_rating:.1f}</span>
                                    <span style='color: #777; font-size: 0.75rem;'>({int(row.num_ratings):,})</span>
                                </div>
                                <div class='card-desc'>{description}</div>
                                <div class='card-genres'>{genres_html}</div>
                            </div>
                        </div>
                    </a>
                    """, unsafe_allow_html=True)
    else:
        st.markdown("<h3 style='margin-bottom: 0.2rem;'>Rekomendasi Lain yang Mungkin Anda Sukai</h3>", unsafe_allow_html=True)
        st.write("Saran film berdasarkan pola kesamaan rating dari pengguna lain di database MovieLens.")
        
        if not is_in_cf:
            st.warning("""
            ⚠️ **Film ini belum populer di database kami.**
            
            Sistem Collaborative Filtering kami memerlukan minimal 4.000 rating untuk memproses rekomendasi yang akurat.
            Silakan beralih ke tab **Rekomendasi Genre Serupa** atau pilih film populer (seperti Toy Story, Inception, Pulp Fiction, dll) untuk melihat hasil collaborative.
            """)
        else:
            with st.spinner("Mencari kecocokan pola rating..."):
                recs_collab = recommender.recommend_by_collaborative(selected_movie_id, item_sim_df, movies_df, top_n)
                
            if recs_collab.empty:
                st.info("Tidak ada rekomendasi collaborative yang ditemukan.")
            else:
                # Tampilkan metrik presisi di UI
                cf_metrics = recommender_metrics.get_recommendation_metrics(recs_collab, movies_df, "Collaborative")
                st.markdown(f"""
                <div style='background: rgba(229, 9, 20, 0.1); border-left: 4px solid #E50914; padding: 1rem; border-radius: 4px; margin-bottom: 1.5rem;'>
                    <span style='color: #fff; font-weight: bold; margin-right: 15px;'>🧪 Uji Presisi Model:</span>
                    <span style='color: #4ade80; font-weight: bold;'>Precision@10: {cf_metrics['precision@10']*100:.1f}%</span> | 
                    <span style='color: #fbbf24; font-weight: bold;'>Diversity Score: {cf_metrics['diversity_score']:.2f}</span> | 
                    <span style='color: #60a5fa; font-weight: bold;'>Avg Quality Rating: {cf_metrics['avg_rating']:.1f}⭐</span>
                </div>
                """, unsafe_allow_html=True)

                with st.spinner("Mengunduh poster film rekomendasi..."):
                    recs_details = load_details_in_parallel(recs_collab)
                    
                cols = st.columns(4)
                for i, row in enumerate(recs_collab.itertuples()):
                    col_target = cols[i % 4]
                    poster_url, description = recs_details[i]
                    
                    # Format rating stars
                    item_stars = "".join(["★" for _ in range(int(round(row.avg_rating)))]) + "".join(["☆" for _ in range(5 - int(round(row.avg_rating)))])
                    match_pct = int(row.similarity * 100)
                    
                    item_genres = row.genres.split('|')[:3]
                    genres_html = "".join([f"<span class='genre-badge'>{g}</span>" for g in item_genres])
                    
                    with col_target:
                        st.markdown(f"""
                        <a href='/?movieId={row.movieId}' target='_self' style='text-decoration: none; color: inherit;'>
                            <div class='movie-card'>
                                <div class='movie-card-img-container'>
                                    <img class='movie-card-img' src='{poster_url}'>
                                </div>
                                <div class='movie-card-content'>
                                    <div class='card-title'>{row.title}</div>
                                    <div class='card-meta'>
                                        <span class='card-match'>{match_pct}% Cocok</span>
                                        <span class='card-year'>{row.year if row.year > 0 else 'N/A'}</span>
                                    </div>
                                    <div class='card-rating-row'>
                                        {item_stars} <span style='color: #cbd5e1; font-weight: bold;'>{row.avg_rating:.1f}</span>
                                        <span style='color: #777; font-size: 0.75rem;'>({int(row.num_ratings):,})</span>
                                    </div>
                                    <div class='card-desc'>{description}</div>
                                    <div class='card-genres'>{genres_html}</div>
                                </div>
                            </div>
                        </a>
                        """, unsafe_allow_html=True)

# ----------------- TAB 3: STATS & INSIGHTS -----------------
with tab3:
    st.markdown("### Statistik Ringkas Data MovieLens 32M")
    
    # Grid of Metrics
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.markdown(f"""
        <div class='metric-box'>
            <div style='font-size: 2rem; font-weight: 800; color: #E50914;'>{len(movies_df):,}</div>
            <div style='font-size: 0.85rem; color: #888; text-transform: uppercase;'>Total Film terindeks</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown("""
        <div class='metric-box'>
            <div style='font-size: 2rem; font-weight: 800; color: #E50914;'>32,000,204</div>
            <div style='font-size: 0.85rem; color: #888; text-transform: uppercase;'>Total rating user</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m3:
        st.markdown(f"""
        <div class='metric-box'>
            <div style='font-size: 2rem; font-weight: 800; color: #E50914;'>{item_sim_df.shape[0]:,}</div>
            <div style='font-size: 0.85rem; color: #888; text-transform: uppercase;'>Film Populer (Model CF)</div>
        </div>
        """, unsafe_allow_html=True)
    with col_m4:
        st.markdown("""
        <div class='metric-box'>
            <div style='font-size: 2rem; font-weight: 800; color: #E50914;'>26,179</div>
            <div style='font-size: 0.85rem; color: #888; text-transform: uppercase;'>User Aktif (Model CF)</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        st.subheader("Distribusi Rilis Film Berdasarkan Dekade")
        valid_years = movies_df[movies_df['year'] > 0]['year']
        
        fig, ax = plt.subplots(figsize=(8, 4), facecolor='#141414')
        ax.set_facecolor('#141414')
        
        sns.histplot(valid_years, bins=30, kde=True, color='#E50914', ax=ax)
        
        # Style adjustments for dark theme
        ax.tick_params(colors='#aaaaaa')
        ax.xaxis.label.set_color('#aaaaaa')
        ax.yaxis.label.set_color('#aaaaaa')
        ax.set_xlabel("Tahun Rilis", fontsize=10)
        ax.set_ylabel("Jumlah Film", fontsize=10)
        ax.spines['bottom'].set_color('#2f2f2f')
        ax.spines['left'].set_color('#2f2f2f')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        
        st.pyplot(fig)
        
    with col_chart2:
        st.subheader("Top 10 Genre Film Paling Dominan")
        all_genres = movies_df['genres'].str.split('|').explode()
        all_genres = all_genres[all_genres != '(no genres listed)']
        genre_counts = all_genres.value_counts().head(10)
        
        fig, ax = plt.subplots(figsize=(8, 4), facecolor='#141414')
        ax.set_facecolor('#141414')
        
        # Use single hue argument to avoid seaborn warnings
        sns.barplot(x=genre_counts.values, y=genre_counts.index, hue=genre_counts.index, palette='rocket', legend=False, ax=ax)
        
        ax.tick_params(colors='#aaaaaa')
        ax.xaxis.label.set_color('#aaaaaa')
        ax.yaxis.label.set_color('#aaaaaa')
        ax.set_xlabel("Jumlah Film", fontsize=10)
        ax.set_ylabel("Genre", fontsize=10)
        ax.spines['bottom'].set_color('#2f2f2f')
        ax.spines['left'].set_color('#2f2f2f')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        
        st.pyplot(fig)
