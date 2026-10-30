@echo off
echo Starting Movie Recommendation System...
python train_model.py
streamlit run app.py
pause
