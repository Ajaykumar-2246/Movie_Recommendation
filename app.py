import streamlit as st
import pandas as pd
import numpy as np
import ast
import requests
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Get the API key securely from Streamlit Secrets
api_key = st.secrets["general"]["TMDB_API_KEY"]

# Title and description of the app
st.title("Movie Recommendation System")
st.write("This app recommends movies based on your preferences, along with movie posters.")

# Function to convert dictionary strings to lists
def convert_to_list(dictionary_words):
    list_words = []
    for i in ast.literal_eval(dictionary_words):
        if 'name' in i:
            list_words.append(i['name'])
    return list_words

# Function to process cast (top 3 actors)
def convert_cast(words):
    lis = []
    count = 0
    for i in ast.literal_eval(words):
        if count != 3:
            if 'name' in i:
                lis.append(i['name'])
            count += 1
        else:
            break
    return lis

# Function to fetch the director
def fetch_director(obj):
    lis = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            lis.append(i['name'])
            break
    return lis

# Load dataset
@st.cache_data
def load_data():
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits = pd.read_csv('tmdb_5000_credits.csv')
    movies_data = movies.merge(credits, on='title')
    movies_data = movies_data[['movie_id', 'title', 'genres', 'overview', 'keywords', 'cast', 'crew']]
    movies_data.dropna(inplace=True)

    movies_data['genres'] = movies_data['genres'].apply(convert_to_list)
    movies_data['keywords'] = movies_data['keywords'].apply(convert_to_list)
    movies_data['cast'] = movies_data['cast'].apply(convert_cast)
    movies_data['crew'] = movies_data['crew'].apply(fetch_director)

    # Process text columns
    movies_data['overview'] = movies_data['overview'].apply(lambda x: [word for word in x.split()])
    movies_data['genres'] = movies_data['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies_data['cast'] = movies_data['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies_data['keywords'] = movies_data['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies_data['crew'] = movies_data['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

    # Create tags
    movies_data['tags'] = movies_data['overview'] + movies_data['genres'] + movies_data['keywords'] + movies_data['cast'] + movies_data['crew']
    new_data = movies_data[['movie_id', 'title', 'tags']]
    new_data['tags'] = new_data['tags'].apply(lambda x: " ".join(x).lower())

    return new_data

movies_df = load_data()

# Text vectorization
@st.cache_data
def create_similarity():
    ps = PorterStemmer()

    def stem(text):
        return " ".join([ps.stem(word) for word in text.split()])

    movies_df['tags'] = movies_df['tags'].apply(stem)
    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(movies_df['tags']).toarray()
    similarity = cosine_similarity(vectors)

    return similarity

similarity_matrix = create_similarity()

# Movie selection and recommendation
movie_titles = movies_df['title'].values
selected_movie = st.selectbox("Select a movie you like:", movie_titles)

# Function to fetch movie poster
def fetch_poster(movie_title):
    url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={movie_title}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if data['results']:
            poster_path = data['results'][0]['poster_path']
            return f"https://image.tmdb.org/t/p/w500{poster_path}"
    return ""

# Recommendation function
def recommend(movie, num_recommendations=10):
    try:
        movie_index = movies_df[movies_df['title'] == movie].index[0]
        distances = list(enumerate(similarity_matrix[movie_index]))
        sorted_movies = sorted(distances, key=lambda x: x[1], reverse=True)[1:num_recommendations + 1]

        recommended_titles = [movies_df.iloc[i[0]]['title'] for i in sorted_movies]
        return recommended_titles
    except IndexError:
        return []

if st.button("Recommend Movies"):
    recommendations = recommend(selected_movie)
    if recommendations:
        st.subheader("Top Recommendations:")
        for title in recommendations:
            poster_url = fetch_poster(title)
            col1, col2 = st.columns([1, 4])
            with col1:
                if poster_url:
                    st.image(poster_url, width=100)
                else:
                    st.write("No Poster Available")
            with col2:
                st.write(title)
    else:
        st.error("No recommendations found. Please try another movie.")
