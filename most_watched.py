import pandas as pd

# Load dataset
splits = {'train': 'data/train-00000-of-00001-8c8c7645a52d95e5.parquet', 'validation': 'data/validation-00000-of-00001-609ec132d91847f9.parquet'}
df = pd.read_parquet("hf://datasets/ashraq/movielens_ratings/" + splits["train"])

# create a system to recommend movies to a user based on the total rating of the movies

# create a system to recommend movies to a user based on the total rating of the movies

def get_top_movies(df, n=10):
    """
    Retourne les n films les plus regardés basés sur le nombre de notes
    """
    # Compter le nombre de notes par film
    movie_counts = df['movie_id'].value_counts()
    
    # Calculer la note moyenne par film
    movie_ratings = df.groupby('title')['rating'].agg(['count', 'mean'])
    
    # Filtrer les films avec un minimum de notes (par exemple, 100)
    min_ratings = 100
    qualified_movies = movie_ratings[movie_ratings['count'] >= min_ratings]
    
    # Trier par nombre de notes puis par note moyenne
    top_movies = qualified_movies.sort_values(['count', 'mean'], ascending=[False, False])
    
    return top_movies.head(n)

def recommend_popular_movies(df, n=10):
    """
    Recommande les n films les plus populaires
    """
    top_movies = get_top_movies(df, n)
    
    # Obtenir les informations détaillées des films
    recommendations = pd.DataFrame(top_movies).reset_index()
    recommendations.columns = ['titre', 'nombre_notes', 'note_moyenne']
    
    return recommendations

def get_top_movies_by_genre(df, genre, n=10):
    """
    Retourne les n films les plus regardés d'un genre spécifique
    """
    # Filtrer les films qui appartiennent au genre spécifié
    genre_movies = df[df['genres'].str.contains(genre, na=False)]
    
    # Calculer la note moyenne par film pour ce genre
    movie_ratings = genre_movies.groupby('title')['rating'].agg(['count', 'mean'])
    
    # Filtrer les films avec un minimum de notes
    min_ratings = 50  # Seuil plus bas car on filtre déjà par genre
    qualified_movies = movie_ratings[movie_ratings['count'] >= min_ratings]
    
    # Trier par nombre de notes puis par note moyenne
    top_movies = qualified_movies.sort_values(['count', 'mean'], ascending=[False, False])
    
    return top_movies.head(n)

def recommend_genre_movies(df, genre, n=10):
    """
    Recommande les n films les plus populaires d'un genre spécifique
    """
    top_genre_movies = get_top_movies_by_genre(df, genre, n)
    
    # Obtenir les informations détaillées des films
    recommendations = pd.DataFrame(top_genre_movies).reset_index()
    recommendations.columns = ['titre', 'nombre_notes', 'note_moyenne']
    recommendations['genres'] = genre
    
    return recommendations

def get_all_genres(df):
    """
    Retourne la liste de tous les genres uniques dans le dataset
    """
    # Concaténer tous les genres et les diviser
    all_genres = df['genres'].str.split('|', expand=True).stack()
    # Retourner les genres uniques triés
    return sorted(all_genres.unique())

# Exemple d'utilisation
if __name__ == "__main__":
    # Afficher tous les genres disponibles
    print("\nGenres disponibles :")
    print(get_all_genres(df))
    for genre in get_all_genres(df) :
        if genre != "(no genres listed)":
            print(f"\nFilms les plus populaires du genre {genre} :")
            print(recommend_genre_movies(df, genre))