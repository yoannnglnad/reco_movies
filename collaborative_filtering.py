import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD

# Load dataset
splits = {'train': 'data/train-00000-of-00001-8c8c7645a52d95e5.parquet', 'validation': 'data/validation-00000-of-00001-609ec132d91847f9.parquet'}
df = pd.read_parquet("hf://datasets/ashraq/movielens_ratings/" + splits["train"])

# Use only a subset for demonstration (first 100,000 rows)
df_10000 = df.head(100000)

# Create the ratings utility matrix
ratings_utility_matrix = df.pivot_table(values='rating', index='user_id', columns='movie_id', fill_value=0)

# Perform SVD
X = ratings_utility_matrix.T  # Transpose to get movies as rows
SVD = TruncatedSVD(n_components=10)
decomposed_matrix = SVD.fit_transform(X)

# Compute correlation matrix
correlation_matrix = np.corrcoef(decomposed_matrix)

# Assume you want recommendations for a specific movie (e.g., movie_id 3)
i = 3

# Get the movie index and its correlation with others
movies_names = list(X.index)
movies_ID = movies_names.index(i)
correlation_movies_ID = correlation_matrix[movies_ID]

# Get recommended movie IDs based on correlation threshold
Recommend = list(X.index[correlation_movies_ID > 0.90])
Recommend.remove(i)  # Remove the movie that was already bought

# Display the recommended movies (top 9)
for movie_id in Recommend[:9]:  # Taking only the first 9 recommendations
    movie_title = df.loc[df["movie_id"] == movie_id, "title"].values
    if len(movie_title) > 0:
        print(f"Recommended Movie: {movie_title[0]}")
    else:
        print(f"Movie ID {movie_id} not found in dataset.")
