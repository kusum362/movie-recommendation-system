from recommendation import MovieRecommender

model = MovieRecommender()
model.df.to_csv("data/movies_with_clusters.csv", index=False)
print("Training completed.")
print("Number of movies:", len(model.df))
print("Clusters:", sorted(model.df["cluster"].unique().tolist()))
