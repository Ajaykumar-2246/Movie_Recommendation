# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import ast
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Title and description of the app
st.title("Movie Recommendation System")
st.write("This app recommends movies based on your preferences, displaying movie posters along with recommendations.")

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

# Load movie data
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

    movies_data['overview'] = movies_data['overview'].apply(lambda x: [word for word in x.split()])
    movies_data['genres'] = movies_data['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies_data['cast'] = movies_data['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies_data['keywords'] = movies_data['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])
    movies_data['crew'] = movies_data['crew'].apply(lambda x: [i.replace(" ", "") for i in x])

    movies_data['tags'] = movies_data['overview'] + movies_data['genres'] + movies_data['keywords'] + movies_data['cast'] + movies_data['crew']
    # Create a new copy to avoid SettingWithCopyWarning
    new_data = movies_data[['movie_id', 'title', 'tags']].copy()

# Safely apply transformations to the 'tags' column
    new_data.loc[:, 'tags'] = new_data['tags'].apply(lambda x: " ".join(x).lower())


    return new_data

movies_df = load_data()

# Load poster data
@st.cache_data
def load_poster_data():
    return pd.read_csv('movie_posters.csv')

poster_data = load_poster_data()

# Function to fetch poster from local dataset
def fetch_poster_local(movie_title):
    poster_row = poster_data[poster_data['title'].str.lower() == movie_title.lower()]
    if not poster_row.empty:
        return poster_row.iloc[0]['poster_url']
    return ""

# Text vectorization
@st.cache_resource
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

# Display recommendations
if st.button("Recommend Movies"):
    recommendations = recommend(selected_movie)
    if recommendations:
        st.subheader("Top Recommendations:")
        for title in recommendations:
            col1, col2 = st.columns([1, 4])  # Adjust column ratio for poster and title
            with col1:
                poster_url = fetch_poster_local(title)
                if poster_url:
                    st.image(poster_url, width=100)  # Adjust image width
            with col2:
                st.write(f"**{title}**")
    else:
        st.error("No recommendations found. Please try another movie.")
