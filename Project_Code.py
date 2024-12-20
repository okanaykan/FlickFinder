import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set display options for pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

# Load datasets
# Dataset 1: Credits
credits_df = pd.read_csv("tmdb_5000_credits.csv")
# Dataset 2: Movies
movies_df = pd.read_csv("tmdb_5000_movies.csv")

# Merge the datasets on 'title'
df = movies_df.merge(credits_df, on='title')

# Select relevant columns
df = df[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Drop missing values
df.dropna(inplace=True)

# Convert 'genres' and 'keywords' columns into a list of names
def convert(text):
    result = []
    for i in ast.literal_eval(text):
        result.append(i['name'])
    return result

df['genres'] = df['genres'].apply(convert)
df['keywords'] = df['keywords'].apply(convert)

# Extract the top 3 cast members
def convert_cast(text):
    cast = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            cast.append(i['name'])
        counter += 1
    return cast

df['cast'] = df['cast'].apply(convert_cast)

# Extract the director from the 'crew' column
def pull_director(text):
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            return [i['name']]
    return []

df['crew'] = df['crew'].apply(pull_director)

# Split 'overview' into a list of words
df['overview'] = df['overview'].apply(lambda x: x.split())

# Remove spaces in the lists
def remove_space(lst):
    return [i.replace(" ", "") for i in lst]

df['cast'] = df['cast'].apply(remove_space)
df['crew'] = df['crew'].apply(remove_space)
df['genres'] = df['genres'].apply(remove_space)
df['keywords'] = df['keywords'].apply(remove_space)

# Combine relevant columns into 'tags'
df['tags'] = df['overview'] + df['genres'] + df['keywords'] + df['cast'] + df['crew']

# Create a new DataFrame with selected columns
movies_df = df[['movie_id', 'title', 'tags']]
movies_df = movies_df.copy()

# Convert 'tags' list to a single string
movies_df['tags'] = movies_df['tags'].apply(lambda x: " ".join(x))

# Convert 'tags' to lowercase
movies_df['tags'] = movies_df['tags'].apply(lambda x: x.lower())

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['tags'].values.astype('U'))

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to get movie recommendations
def get_recommendations(title, cosine_sim=cosine_sim):
    # Get the index of the movie with the given title
    idx = movies_df[movies_df['title'] == title].index[0]

    # Get similarity scores for all movies
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort movies by similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the top 10 similar movies
    sim_scores = sim_scores[1:11]

    # Get movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the titles of the recommended movies
    return movies_df['title'].iloc[movie_indices]

# Example: Get recommendations for a given movie title
film_title = "Total Recall"
print("Movie Recommendations:")
print(get_recommendations(film_title))























