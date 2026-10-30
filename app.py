import streamlit as st
from recommendation import MovieRecommender

st.set_page_config(page_title="Movie Recommendation System", page_icon="🎬", layout="wide")
st.title("🎬 Movie Recommendation System")
st.write("Machine Learning project using TF-IDF, kNN, K-Means and similarity measures.")

@st.cache_resource
def load_model():
    return MovieRecommender()

model = load_model()

tab1, tab2, tab3 = st.tabs(["🎯 Similar Movies", "⭐ Preference Based", "📊 Movie Clusters"])

with tab1:
    st.subheader("Find movies similar to a selected movie")
    title = st.selectbox("Choose a movie", model.titles())
    n = st.slider("Number of recommendations", 3, 10, 5)
    if st.button("Recommend Similar Movies"):
        result = model.knn_recommend(title, n)
        st.dataframe(result, use_container_width=True)

with tab2:
    st.subheader("Recommend movies from your desired experience")
    preference = st.text_input(
        "Enter preferences",
        "action suspense technology"
    )
    n2 = st.slider("Number of preference recommendations", 3, 10, 5)
    if st.button("Generate Personalized Recommendations"):
        result = model.preference_recommend(preference, n2)
        st.dataframe(result, use_container_width=True)

with tab3:
    st.subheader("K-Means movie groups")
    cluster = st.selectbox("Select cluster", sorted(model.df["cluster"].unique()))
    st.dataframe(model.cluster_movies(cluster), use_container_width=True)

st.divider()
st.caption("ML concepts: preprocessing • TF-IDF • cosine similarity • kNN • K-Means • ranking")
