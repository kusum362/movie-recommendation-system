import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "data/movies.csv"

class MovieRecommender:
    def __init__(self, data_path=DATA_PATH):
        self.df = pd.read_csv(data_path)
        self.df["combined"] = (
            self.df["genres"].fillna("") + " " +
            self.df["keywords"].fillna("") + " " +
            self.df["cast"].fillna("") + " " +
            self.df["director"].fillna("")
        )
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf = self.vectorizer.fit_transform(self.df["combined"])
        self.knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=6)
        self.knn.fit(self.tfidf)

        numeric = self.df[["rating","popularity"]].copy()
        self.scaler = StandardScaler()
        numeric_scaled = self.scaler.fit_transform(numeric)
        self.kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        self.df["cluster"] = self.kmeans.fit_predict(numeric_scaled)

    def titles(self):
        return self.df["title"].tolist()

    def knn_recommend(self, title, n=5):
        idx = self.df.index[self.df["title"].str.lower() == title.lower()]
        if len(idx) == 0:
            raise ValueError("Movie not found.")
        idx = idx[0]
        distances, indices = self.knn.kneighbors(self.tfidf[idx], n_neighbors=min(n+1, len(self.df)))
        rows = []
        for distance, i in zip(distances[0], indices[0]):
            if i != idx:
                rows.append({
                    "title": self.df.iloc[i]["title"],
                    "similarity": round(1-distance, 3),
                    "rating": self.df.iloc[i]["rating"],
                    "genres": self.df.iloc[i]["genres"]
                })
        return pd.DataFrame(rows).head(n)

    def preference_recommend(self, preference, n=5):
        query = self.vectorizer.transform([preference])
        sims = cosine_similarity(query, self.tfidf).flatten()
        result = self.df.copy()
        result["similarity"] = sims
        result["score"] = 0.75*result["similarity"] + 0.25*(result["rating"]/10)
        return result.sort_values("score", ascending=False).head(n)[
            ["title","genres","rating","similarity","score","cluster"]
        ]

    def cluster_movies(self, cluster_id):
        return self.df[self.df["cluster"] == cluster_id][
            ["title","genres","rating","popularity","cluster"]
        ].sort_values("rating", ascending=False)
