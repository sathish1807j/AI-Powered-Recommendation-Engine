from src.data import prepare_data

train, test, movies, num_users, num_movies, user2index, movie2index = prepare_data()

print(f"Number of users: {num_users}")
print(f"Number of movies: {num_movies}")
print(f"Train dataset shape: {train.shape}")
print(f"Test dataset shape: {test.shape}")
print(train.head())
