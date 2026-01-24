import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data():
    # Load datasets
    ratings = pd.read_csv("data/ratings.csv")   # userId, movieId, rating, timestamp
    movies = pd.read_csv("data/movies.csv")     # movieId, title

    # Encode users and movies as indices
    user2idx = {uid: idx for idx, uid in enumerate(ratings['userId'].unique())}
    movie2idx = {mid: idx for idx, mid in enumerate(ratings['movieId'].unique())}

    ratings['userIndex'] = ratings['userId'].map(user2idx)
    ratings['movieIndex'] = ratings['movieId'].map(movie2idx)

    # Add movieIndex to movies DataFrame
    movies['movieIndex'] = movies['movieId'].map(movie2idx)

    # Train-test split
    train_df, test_df = train_test_split(ratings, test_size=0.2, random_state=42)

    num_users = ratings['userIndex'].nunique()
    num_movies = ratings['movieIndex'].nunique()

    return train_df, test_df, num_users, num_movies, movies
