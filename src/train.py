import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.data import prepare_data
from src.model import MatrixFactorization

# -----------------------------
# Dataset class
# -----------------------------
class RatingsDataset(Dataset):
    def __init__(self, df):
        self.users = torch.tensor(df['userIndex'].values, dtype=torch.long)
        self.movies = torch.tensor(df['movieIndex'].values, dtype=torch.long)
        self.ratings = torch.tensor(df['rating'].values, dtype=torch.float)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.ratings[idx]

# -----------------------------
# Training function
# -----------------------------
def train():
    train_df, test_df, num_users, num_movies, _ = prepare_data()
    
    train_dataset = RatingsDataset(train_df)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    
    model = MatrixFactorization(num_users, num_movies)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    
    epochs = 5
    for epoch in range(epochs):
        total_loss = 0
        for users, movies, ratings in train_loader:
            optimizer.zero_grad()
            preds = model(users, movies)
            loss = loss_fn(preds, ratings)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(ratings)
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_dataset):.4f}")

    torch.save(model.state_dict(), "model.pth")
    print("Model saved as model.pth")

# -----------------------------
# Run training
# -----------------------------
if __name__ == "__main__":
    train()
