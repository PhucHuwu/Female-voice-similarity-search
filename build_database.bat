@echo off
REM Master script to build the entire voice similarity search database
REM Run this after downloading YouTube videos to data/raw/

echo ============================================================
echo Voice Similarity Search - Database Builder
echo ============================================================
echo.

REM Activate conda environment
call conda activate voice-search
if errorlevel 1 (
    echo Error: Failed to activate conda environment 'voice-search'
    echo Please run: conda env create -f environment.yml
    pause
    exit /b 1
)

echo Step 1/3: Splitting YouTube videos into 3s chunks...
echo ============================================================
python src/data_collection/split_audio_chunks.py
if errorlevel 1 (
    echo Error in chunking step
    pause
    exit /b 1
)

echo.
echo Step 2/3: Preprocessing chunks (16kHz, normalized)...
echo ============================================================
python src/data_collection/preprocess_audio.py
if errorlevel 1 (
    echo Error in preprocessing step
    pause
    exit /b 1
)

echo.
echo Step 3/3: Building FAISS database...
echo ============================================================
python scripts/build_database.py
if errorlevel 1 (
    echo Error in database build step
    pause
    exit /b 1
)

echo.
echo ============================================================
echo SUCCESS! Database is ready
echo ============================================================
echo.
echo Next: Run the Streamlit app
echo   streamlit run app/streamlit_app.py
echo.
pause
