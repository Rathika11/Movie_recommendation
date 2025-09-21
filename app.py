import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# Title
st.title("🎬 Movie Recommendation System")

# Upload dataset
uploaded_file = st.file_uploader("Upload a Movies Dataset (CSV)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # Check required columns
    if 'title' in df.columns and 'genres' in df.columns:
        # Text vectorization
        cv = CountVectorizer(stop_words='english')
        vectors = cv.fit_transform(df['genres'].fillna(''))
        similarity = cosine_similarity(vectors)

        # Movie search
        movie_name = st.selectbox("🎥 Choose a Movie:", df['title'].values)

        if st.button("🔍 Recommend"):
            index = df[df['title'] == movie_name].index[0]
            distances = list(enumerate(similarity[index]))
            sorted_movies = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]

            st.subheader("✨ Recommended Movies:")
            for i in sorted_movies:
                st.write(f"🎞️ {df.iloc[i[0]].title}  ({df.iloc[i[0]].genres})")

    else:
        st.error("Dataset must contain at least 'title' and 'genres' columns.")
else:
    st.info("Please upload a CSV file with movies and genres.")
