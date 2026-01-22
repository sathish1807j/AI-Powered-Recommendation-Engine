import pandas as pd
from sklearn.model_selection import train_test_split

def prepare_data(ratings_path="data/ratings.csv", movies_path="data/movies.csv", test_size=0.2, random_state=42):
    # Load CSVs
    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    
    # Map userId and movieId to continuous indexes
    user_ids = ratings['userId'].unique()
    movie_ids = ratings['movieId'].unique()
    user2index = {uid: idx for idx, uid in enumerate(user_ids)}
    movie2index = {mid: idx for idx, mid in enumerate(movie_ids)}
    
    ratings['userIndex'] = ratings['userId'].map(user2index)
    ratings['movieIndex'] = ratings['movieId'].map(movie2index)
    
    # Split into train and test
    train, test = train_test_split(ratings, test_size=test_size, random_state=random_state)
    
    num_users = len(user_ids)
    num_movies = len(movie_ids)
    
    return train, test, movies, num_users, num_movies, user2index, movie2index
