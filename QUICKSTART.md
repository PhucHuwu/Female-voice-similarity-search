# Quick Start Scripts

## setup.bat - Windows Batch Script

Run this to setup the entire system automatically.

```batch
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
echo 1. Download data: python src/data_collection/download_audio.py
echo 2. Preprocess: python src/data_collection/preprocess_audio.py
echo 3. Build database: python scripts/build_database.py
echo 4. Run app: streamlit run app/streamlit_app.py
echo.
pause
```

## run_pipeline.bat - Run Complete Pipeline

```batch
@echo off
call conda activate voice-search

echo ========================================
echo Running Complete Pipeline
echo ========================================

echo.
echo [1/3] Downloading sample dataset...
python src/data_collection/download_audio.py

echo.
echo [2/3] Preprocessing audio files...
python src/data_collection/preprocess_audio.py

echo.
echo [3/3] Building FAISS database...
python scripts/build_database.py

echo.
echo ========================================
echo Pipeline complete!
echo ========================================
echo.
echo Starting Streamlit app...
streamlit run app/streamlit_app.py

pause
```

## For Manual Step-by-Step:

### 1. Setup Environment

```bash
conda env create -f environment.yml
conda activate voice-search
```

### 2. Download Data (50 samples for testing)

```bash
python src/data_collection/download_audio.py
```

### 3. Preprocess Audio

```bash
python src/data_collection/preprocess_audio.py
```

### 4. Build Database

```bash
python scripts/build_database.py
```

### 5. Run Tests (optional)

```bash
python tests/run_tests.py
```

### 6. Launch App

```bash
streamlit run app/streamlit_app.py
```

Visit: http://localhost:8501
