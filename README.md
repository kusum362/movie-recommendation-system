# Movie Recommendation System using Machine Learning

This project is based on the uploaded ML review presentation. It demonstrates:
- Movie feature preprocessing
- TF-IDF feature extraction
- Cosine similarity / distance calculation
- k-Nearest Neighbour (kNN) for finding similar movies
- K-Means clustering for grouping movies
- Preference-based recommendation
- Ranked recommendations
- Simple Streamlit interface

## Project structure
- `app.py` - Streamlit user interface
- `recommendation.py` - ML recommendation functions
- `train_model.py` - trains/saves K-Means and kNN artifacts
- `data/movies.csv` - self-contained sample dataset
- `notebooks/Movie_Recommendation.ipynb` - step-by-step notebook
- `requirements.txt` - required Python packages
- `run_project.bat` - Windows one-click launcher after dependencies are installed

## Run in Anaconda
1. Open Anaconda Prompt.
2. Create environment:
   `conda create -n movie_ml python=3.10 -y`
3. Activate:
   `conda activate movie_ml`
4. Go to this project folder.
5. Install:
   `pip install -r requirements.txt`
6. Train the model:
   `python train_model.py`
7. Start the app:
   `streamlit run app.py`

## What to explain in review
kNN finds movies close to a selected movie using feature distance.
K-Means groups movies into clusters with similar characteristics.
TF-IDF converts text features such as genres and keywords into numerical vectors.
Cosine similarity measures how close two movie vectors are.
The final recommender ranks candidate movies by similarity and user preferences.
