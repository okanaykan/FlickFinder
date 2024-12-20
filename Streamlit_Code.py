import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
import ast

## Main Page Tasarımı ##


st.set_page_config(layout="wide", page_title="FlickFinder", page_icon="🍿")
st.title(":blue[FlickFinder] 🎥✨")
st.subheader(":red[Your Personalized Movie Guide!]")
st.markdown("""
<style>

.stButton > button {
     background-color: #FF6347;  /* Daha canlı bir renk tonu */
    color: white;
    border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)
home_tab, recommendation_tab, data_tab = st.tabs(["Home", "Find Your Next Favorite", "Data"])
text_left, image_middle, text_right = home_tab.columns([1, 1, 1], gap="medium")
with text_left:
    st.markdown("""
    <div style="text-align: left; font-size: 18px; line-height: 1.6;">
    Welcome to <span style="font-weight: bold; font-style: italic;">FlickFinder</span>, your trusted guide to the world of cinema!<br>
    Whether you’re a casual viewer or a film aficionado, we’re here to help you find movies tailored to your unique taste.
    </div>
    """, unsafe_allow_html=True)

with image_middle:
    st.image("5.png", caption="Film App", use_container_width=True)

with text_right:
    st.markdown("""
    <div style="text-align: left; font-size: 18px; line-height: 1.6;">
    🎬 <strong>Discover:</strong> Dive into a curated selection of films with the help of our advanced algorithms and real user reviews.<br><br>
    📚 <strong>Explore:</strong> From blockbuster hits to hidden gems, your next favorite movie is just a click away.<br><br>
    Let FlickFinder be your partner in uncovering cinematic treasures. Click the "<em>Find Your Next Favorite</em>" tab above to start your journey!
    </div>
    """, unsafe_allow_html=True)

st.logo(
        "3.png",
        size="large",
        icon_image="4.png")
st.sidebar.title(":red[Discover Your Perfect Movie!] 🎥")
st.sidebar.markdown("Explore the world of cinema and find films you'll love.")
st.sidebar.image("2.png", caption="Uncover cinematic gems for an unforgettable experience.",use_container_width=True)
st.sidebar.markdown("### Popular Genres")
st.sidebar.markdown("""
- **Action** - Thrilling adventures and epic battles.
- **Comedy** - Laugh out loud with hilarious stories.
- **Drama** - Emotional and captivating tales.
- **Romance** - Heartwarming love stories.
- **Sci-Fi** - Dive into futuristic worlds and ideas.
- **Horror** - Spine-chilling and eerie experiences.
""")

st.sidebar.markdown("### About")
st.sidebar.info(
            "This movie recommendation system helps you discover films tailored to your preferences. "
            "Use the filters to explore genres, directors, and ratings, and uncover your next favorite movie!"
        )

## Recommendation Tab ##
@st.cache_data
def get_data():
    credits_df = pd.read_csv("tmdb_5000_credits.csv")
    movies_df = pd.read_csv("tmdb_5000_movies.csv")
    df = movies_df.merge(credits_df, on='title')
    df = df[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    df.dropna(inplace=True)

    def convert(text):
        return [i['name'] for i in ast.literal_eval(text)]

    df['genres'] = df['genres'].apply(convert)
    df['keywords'] = df['keywords'].apply(convert)

    def convert_cast(text):
        return [i['name'] for i in ast.literal_eval(text)[:3]]

    df['cast'] = df['cast'].apply(convert_cast)

    def pull_director(text):
        return [i['name'] for i in ast.literal_eval(text) if i['job'] == 'Director']

    df['crew'] = df['crew'].apply(pull_director)
    df['overview'] = df['overview'].apply(lambda x: x.split())

    def remove_space(lst):
        return [i.replace(" ", "") for i in lst]

    for col in ['genres', 'keywords', 'cast', 'crew']:
        df[col] = df[col].apply(remove_space)

    df['cast'] = df['cast'].apply(remove_space)
    df['crew'] = df['crew'].apply(remove_space)
    df['genres'] = df['genres'].apply(remove_space)
    df['keywords'] = df['keywords'].apply(remove_space)

    df['tags'] = df['overview'].apply(lambda x: " ".join(x)) + " " + \
                 df['genres'].apply(lambda x: " ".join(x)) + " " + \
                 df['keywords'].apply(lambda x: " ".join(x)) + " " + \
                 df['cast'].apply(lambda x: " ".join(x)) + " " + \
                 df['crew'].apply(lambda x: " ".join(x))
    Movie_df = df[['movie_id', 'title', 'tags']]
    return Movie_df

# TF-IDF vectorization
@st.cache_data
def create_tfidf_matrix(data):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['tags'].values.astype('U'))
    return tfidf_matrix

# Compute cosine similarity
@st.cache_data
def calculate_cosine_similarity(_tfidf_matrix):
    cosine_sim = cosine_similarity(_tfidf_matrix, _tfidf_matrix)
    return cosine_sim

# Function to get movie recommendations
@st.cache_data
def get_recommendations(title, data, cosine_sim):
    idx = data[data['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    recommended_movies = [data['title'].iloc[i[0]] for i in sim_scores]
    return recommended_movies

Movie_df = get_data()
tfidf_matrix = create_tfidf_matrix(Movie_df)
cosine_sim = calculate_cosine_similarity(tfidf_matrix)

# Movie poster
def fetch_poster(movie_title):
    url = "https://api.themoviedb.org/3/search/movie"
    params = {
        'api_key': 'daea96ea7d19313e476f678148d8f521',
        'query': movie_title
    }
    response = requests.get(url, params=params)
    data = response.json()

    if 'results' in data and len(data['results']) > 0:
        movie = data['results'][0]
        title = movie['title']
        overview = movie['overview']
        poster_path = movie.get('poster_path', None)
        poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else None

        return {
            'title': movie['title'],
            'overview': movie['overview'],
            'poster_url': poster_url
        }
    return None

# Movie Selection Interface
movie_title = recommendation_tab.selectbox('**Find Movie Recommendations Similar To:**', Movie_df['title'].values)

if recommendation_tab.button(' Get a Recommendation', icon="🎞️"):
    recommendation_tab.markdown("""
    For more details about movies, visit [IMDB's homepage](https://www.imdb.com).
    """,unsafe_allow_html=True)
    # Get recommendations
    recommendations = get_recommendations(movie_title, Movie_df, cosine_sim)

    # Display recommended movies
    recommendation_tab.subheader('Recommended Movies')
    cols = recommendation_tab.columns(len(recommendations))  # Create columns
    for i, recommendation in enumerate(recommendations):
        with cols[i]:  # Add each movie and poster to a column
            poster_data = fetch_poster(recommendation)
            if poster_data and poster_data['poster_url']:
                st.image(poster_data['poster_url'], caption=recommendation, width=200, use_container_width=True)
            else:
                st.write(f"Poster not found for {recommendation}.")


## DataTab ##

    data_tab.title("Data")
    data_tab.subheader("DataSet Link:")
    data_tab.markdown("""
        For details of the dataset, visit [Here](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata/data).
        """, )

    with data_tab.expander("Dataset Summary"):
        st.write(f"**Total Movies:** {Movie_df.shape[0]}")
        st.write(f"**Columns:** {', '.join(Movie_df.columns)}")
        st.dataframe(Movie_df.head(10))

    data_tab.markdown(""" FlickFinder Recommendation System

FlickFinder uses an advanced recommendation system to provide users with the most suitable movie suggestions. 
This system measures the similarities between movies to identify content that may interest the user.

Here’s how the process works:

**1. TF-IDF (Term Frequency-Inverse Document Frequency) Method**

To gather information about movies, summaries, genres, cast, and other keywords are combined to create unique "tags" for each film. However, some words are used so frequently that they are less meaningful. TF-IDF is used to make these tags more meaningful and distinctive:

TF (Term Frequency): Measures how often a specific word appears in a movie’s tag.
IDF (Inverse Document Frequency): Evaluates how common or rare a word is across all movies. Words that appear frequently are given less weight, while rare words are given more significance.
As a result, common but less informative words have a lower impact, while rarer and more specific words hold more weight.

**2.Cosine Similarity:**

Once the TF-IDF vectors are created, Cosine Similarity is applied to measure the similarity between movies. This method calculates the angle between two movie vectors. A smaller angle indicates higher similarity:

A value close to 1: The movies are very similar.
A value close to 0: The movies are very different.
For example, if an action-sci-fi movie is selected, the algorithm will recommend other movies with similar action and sci-fi elements.

**3. User-Focused Recommendation Mechanism**
When a user selects a movie, the algorithm retrieves the TF-IDF vector for that movie and compares it with all other movies. The movies with the highest similarity scores are then ranked and recommended to the user.

**4. Dynamic and Personalized Recommendations**
The algorithm can continuously update to find movies that match users' preferences. For instance, as more movies with different genres or themes are tagged, the system can offer a broader range of suggestions.
    """)
