import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
from src.data import prepare_data
from src.model import MatrixFactorization
import re

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="AI-Recommendation Engine",
    page_icon="🎬",
    layout="centered"
)

# -----------------------------
# Custom UI styling (Dark Cards)
# -----------------------------
st.markdown("""
<style>
    /* Main title styling */
    .main-title {
        font-size: 42px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0px;
        background: -webkit-linear-gradient(#ff4b4b, #ff8a8a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-text {
        text-align: center;
        color: #eee;
        margin-bottom: 30px;
    }
    /* Card-like containers for recommendations - dark background for visibility */
    .rec-card {
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
        background-color: #1f1f1f; /* Dark background */
        color: white;              /* White text */
        margin-bottom: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Load data & model
# -----------------------------
@st.cache_resource
def load_resources():
    train_df, test_df, num_users, num_movies, movies = prepare_data()
    model = MatrixFactorization(num_users, num_movies)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    return train_df, test_df, num_users, num_movies, movies, model

train_df, test_df, num_users, num_movies, movies, model = load_resources()

# -----------------------------
# Initialize dynamic new user
# -----------------------------
new_user_id = num_users
if 'user_initialized' not in st.session_state:
    with torch.no_grad():
        new_user_vec = torch.mean(model.user_embedding.weight, dim=0, keepdim=True)
        model.user_embedding.weight = nn.Parameter(
            torch.cat([model.user_embedding.weight, new_user_vec], dim=0)
        )
    st.session_state.user_initialized = True

# -----------------------------
# Header Section
# -----------------------------
st.markdown("<h1 class='main-title'>AI - Recommendation Engine</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-text'>Recommend products, movies, or courses based on user behavior.. Tell us what you like!</p>", unsafe_allow_html=True)

# -----------------------------
# Filter movies by year
# -----------------------------
st.subheader(" Filter movies before selecting")

# Extract year if missing
if 'year' not in movies.columns:
    def extract_year(title):
        match = re.search(r'\((\d{4})\)', title)
        return int(match.group(1)) if match else None
    movies['year'] = movies['title'].apply(extract_year)

movies['year'] = pd.to_numeric(movies['year'], errors='coerce').astype('Int64')
movies = movies.dropna(subset=['year'])

# Year slider
year_min, year_max = int(movies["year"].min()), int(movies["year"].max())
selected_year = st.slider("Filter by release year:", year_min, year_max, (year_min, year_max))

# Apply year filter
filtered_movies = movies[
    (movies["year"] >= selected_year[0]) & (movies["year"] <= selected_year[1])
]

# Update movie titles for multiselect
movie_titles = filtered_movies["title"].tolist()

# -----------------------------
# User Input Section
# -----------------------------
st.subheader("1. Tell us your taste")
selected_movies = st.multiselect(
    "Choose movies you've seen:", 
    movie_titles, 
    placeholder="Search for a movie...",
    max_selections=10
)

user_ratings = {}

if selected_movies:
    st.write("#### 2. Rate them")
    cols = st.columns(2)
    for i, title in enumerate(selected_movies):
        movie_index = int(movies[movies["title"] == title]["movieIndex"].values[0])
        col = cols[i % 2]  # alternate columns
        with col:
            rating = st.select_slider(
                f"{title}",
                options=[1, 2, 3, 4, 5],
                value=3,
                key=f"rating_{movie_index}"
            )
            user_ratings[movie_index] = rating

# -----------------------------
# Recommendation Button
# -----------------------------
st.markdown("---")
if st.button(" Generate My Recommendations"):
    if not user_ratings:
        st.warning("Please rate at least one movie first!")
    else:
        with st.spinner('Calculating your cinematic profile...'):
            # Adjust new user embedding
            rated_movies = torch.tensor(list(user_ratings.keys()), dtype=torch.long)
            ratings_tensor = torch.tensor(list(user_ratings.values()), dtype=torch.float)
            user_tensor = torch.tensor([new_user_id] * len(rated_movies))

            with torch.no_grad():
                preds = model(user_tensor, rated_movies)
                diff = ratings_tensor - preds
                adjustment = (diff.unsqueeze(1) * model.movie_embedding(rated_movies)).mean(0)
                model.user_embedding.weight[new_user_id] += adjustment

            all_movies = torch.arange(num_movies)
            user_tensor_all = torch.tensor([new_user_id] * num_movies)

            with torch.no_grad():
                predictions = model(user_tensor_all, all_movies)

            for m in user_ratings.keys():
                predictions[m] = -1

            top_movies = torch.topk(predictions, 5).indices.numpy()

            st.success("Analysis complete! Here are your top picks:")
            
            # Show recommendations in dark cards with genre and director
            for idx in top_movies:
                row = movies[movies["movieIndex"] == idx].iloc[0]
                genres = ", ".join(row["genres"]) if isinstance(row["genres"], list) else row["genres"]
                director = row.get("director", "Unknown")
                st.markdown(f"""
                <div class="rec-card">
                    <span style="font-size: 1.2rem;">🎥 <b>{row['title']}</b></span><br>
                    Genres: {genres}<br>
                </div>
                """, unsafe_allow_html=True)
