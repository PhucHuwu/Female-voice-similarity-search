@echo off
echo ========================================
echo Voice Similarity Search - Quick Setup
echo ========================================

echo.
echo Step 1: Creating Conda environment...
call conda env create -f environment.yml
if errorlevel 1 (
    echo Error creating conda environment
    pause
    exit /b 1
)

echo.
echo Step 2: Activating environment...
call conda activate voice-search

echo.
echo Step 3: Installing additional dependencies...
pip install -r requirements.txt

echo.
echo ========================================
echo Environment setup complete!
echo ========================================
echo.
echo Next steps:
echo 1. Download data: python src\data_collection\download_audio.py
echo 2. Preprocess: python src\data_collection\preprocess_audio.py
echo 3. Build database: python scripts\build_database.py
echo 4. Run app: streamlit run app\streamlit_app.py
echo.
pause
