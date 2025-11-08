import streamlit as st
import pickle
import pandas as pd

def recommend(movie_title, movies_df, similarity_matrix):
    """Recommends top 10 similar movies based on content similarity."""
    try:
        # Create a mapping from movie title to its index
        indices = pd.Series(movies_df.index, index=movies_df['title']).drop_duplicates()
        
        # Get the index of the movie that matches the title
        idx = indices[movie_title]
    except KeyError:
        return [f"Sorry, the movie '{movie_title}' was not found in the dataset."]

    # Get the similarity scores of all movies with the input movie
    sim_scores = list(enumerate(similarity_matrix[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the top 10 most similar movies (ignore the first one)
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the titles of the top 10 most similar movies
    return movies_df['title'].iloc[movie_indices].tolist()

# --- Load the saved data ---
try:
    # Load the movie dictionary
    movies_dict = pickle.load(open('movies_dict.pkl', 'rb'))
    # Convert the dictionary back to a DataFrame
    movies_df = pd.DataFrame(movies_dict)

    # Load the similarity matrix
    similarity_matrix = pickle.load(open('similarity.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model files not found. Please make sure 'movies_dict.pkl' and 'similarity.pkl' are in the same folder as app.py")
    st.stop()
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()


# --- Streamlit Web App UI ---

st.title('Indian Movie Recommender')
st.markdown('Find movies similar to the one you like!')

# Create a select box for movie selection
movie_list = movies_df['title'].values
selected_movie = st.selectbox(
    'Select a movie from the dropdown:',
    movie_list
)

# Create a button to trigger the recommendation
if st.button('Recommend'):
    if selected_movie:
        st.write(f"Finding movies similar to **{selected_movie}**...")
        recommendations = recommend(selected_movie, movies_df, similarity_matrix)
        
        # Display the recommendations
        st.subheader('Here are your recommendations:')
        for i, movie in enumerate(recommendations):
            st.write(f"{i+1}. {movie}")
    else:
        st.write("Please select a movie.")
