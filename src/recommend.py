import torch
import torch.nn as nn
from src.data import prepare_data
from src.model import MatrixFactorization

import random

def dynamic_user_recommendation():
    
    # Load data
    
    train_df, test_df, num_users, num_movies, movies = prepare_data()

   
    # Load trained model
    
    model = MatrixFactorization(num_users, num_movies)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

   
    # Add new user dynamically
  
    new_user_id = num_users
    with torch.no_grad():
        # Initialize new user embedding as average of existing users
        new_user_vec = torch.mean(model.user_embedding.weight, dim=0, keepdim=True)
        model.user_embedding.weight = nn.Parameter(
            torch.cat([model.user_embedding.weight, new_user_vec], dim=0)
        )

    
    # Show 10 random movies for rating
    
    sample_movies = movies.sample(10)
    print("\nHere are some movies you can rate:\n")
    for _, row in sample_movies.iterrows():
        print(f"Index: {int(row['movieIndex'])} | Title: {row['title']}")

   
    # Input user ratings
    
    user_ratings = {}
    print("\nEnter ratings for movies (1–5). Type -1 to stop.\n")
    while True:
        try:
            movie_index = int(input("Enter movieIndex (-1 to stop): "))
        except ValueError:
            print("Please enter a valid number.")
            continue

        if movie_index == -1:
            break
        if movie_index < 0 or movie_index >= num_movies:
            print("Invalid index. Try again.")
            continue

        try:
            rating = float(input("Enter rating (1–5): "))
        except ValueError:
            print("Enter a valid rating number.")
            continue
        user_ratings[movie_index] = rating

    if not user_ratings:
        print("No ratings provided. Exiting.")
        return

    
    # Adjust new user embedding based on ratings
    
    rated_movies = torch.tensor(list(user_ratings.keys()), dtype=torch.long)
    ratings_tensor = torch.tensor(list(user_ratings.values()), dtype=torch.float)
    user_tensor = torch.tensor([new_user_id] * len(rated_movies))

    with torch.no_grad():
        # Predicted ratings for rated movies
        preds = model(user_tensor, rated_movies)
        # Compute adjustment
        diff = ratings_tensor - preds
        adjustment = (diff.unsqueeze(1) * model.movie_embedding(rated_movies)).mean(0)
        model.user_embedding.weight[new_user_id] += adjustment

    
    # Predict ratings for all movies
    
    all_movies = torch.arange(num_movies)
    user_tensor_all = torch.tensor([new_user_id] * num_movies)

    with torch.no_grad():
        predictions = model(user_tensor_all, all_movies)

    # Exclude already rated movies
    for m in user_ratings.keys():
        predictions[m] = -1

    
    # Top 5 recommendations
    
    top_movies = torch.topk(predictions, 5).indices.numpy()
    print("\n🎯 Top 5 Recommended Movies for You:\n")
    for idx in top_movies:
        title = movies[movies["movieIndex"] == idx]["title"].values[0]
        print(title)



# Run dynamic recommendation

if __name__ == "__main__":
    dynamic_user_recommendation()
