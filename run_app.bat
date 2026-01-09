@echo off
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Starting Anemia Detection Web App...
echo The app will run at: http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

python app.py
