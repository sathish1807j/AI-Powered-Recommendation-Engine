
## 🎬 AI-Powered Movie Recommendation System

AI-Powered Movie Recommendation System is an end-to-end movie recommendation system that provides personalized movie suggestions based on user preferences and ratings. The system uses **Collaborative Filtering with Deep Learning (Matrix Factorization)** and an interactive **Streamlit** frontend to deliver real-time recommendations.

This project demonstrates practical application of machine learning, deep learning, and full-stack integration using Python.

---

🚀 Demo

🎥 Streamlit-based interactive UI
Users select movies they have watched, rate them, and instantly receive AI-generated movie recommendations.
--
## 🚀 Features
- Personalized movie recommendations
- Collaborative filtering using Matrix Factorization
- Deep learning model built with PyTorch
- Cold-start handling for new users
- Interactive Streamlit web interface
- Real-time recommendations
- Optional movie poster integration using TMDB API
- Clean and modular project structure

---

## 🧠 Recommendation Method

The recommendation engine is based on **Collaborative Filtering**:

- Users and movies are represented as low-dimensional embedding vectors
- The model learns latent features from user–movie rating interactions
- Predictions are generated using dot-product of user and movie embeddings
- New users are handled by initializing embeddings using the mean of existing users and dynamically updating based on ratings

---

## 🏗️ System Architecture

User rates movies in Streamlit UI  
→ Ratings sent to recommendation logic  
→ PyTorch model predicts unseen movie ratings  
→ Top-N recommended movies displayed to user  

---

## 📊 Dataset

**MovieLens Dataset**

Files used:
- `ratings.csv`  
  Contains userId, movieId, rating
- `movies.csv`  
  Contains movieId and title

Why MovieLens:
- Real-world dataset
- Widely used for recommendation systems
- Clean and well-structured

---

## 🧰 Tech Stack

**Frontend**
- Streamlit

**Backend / ML**
- Python
- PyTorch
- Pandas
- NumPy

**Dataset**
- MovieLens

---
## 📁 Project Structure





