# Viva / Review Notes

## Why Machine Learning?
The system learns patterns from movie features and uses numerical similarity to recommend related movies.

## Why kNN?
kNN finds the nearest movie vectors to the selected movie. It is simple and suitable for similarity-based recommendation.

## Why K-Means?
K-Means groups movies into clusters based on their numerical feature representation. This helps identify groups of movies with similar characteristics.

## What is TF-IDF?
TF-IDF converts text into numerical vectors. Words that are useful for distinguishing movies receive higher importance.

## What is cosine similarity?
It measures the angle-based similarity between two vectors. A higher similarity means the movie descriptions/features are more alike.

## Input
A selected movie or a user's desired experience such as "action suspense technology".

## Output
A ranked list of recommended movies with similarity/ranking information.

## Main ML pipeline
Data -> preprocessing -> feature extraction -> similarity/model -> ranking -> recommendation -> evaluation.
