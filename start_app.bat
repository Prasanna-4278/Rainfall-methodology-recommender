@echo off
title Rainfall Methodology Recommendation System
echo ========================================================
echo Starting Rainfall Methodology Recommender Web Application...
echo ========================================================
cd /d "C:\Users\paloj\.gemini\antigravity\scratch\rainfall_methodology_recommender"
echo Opening web browser to http://127.0.0.1:8000 ...
start "" "http://127.0.0.1:8000"
echo Starting FastAPI server...
python app.py
pause
