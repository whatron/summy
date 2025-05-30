# Summy - Audio Summarizer


A local-first, real-time audio transcription and summarization app built with Tauri, React, and Python.

## Overview

Summy is a local, privacy-respecting tool for real-time audio transcription and summarization. It captures audio from system and microphone inputs, transcribes it using a locally running Whisper model, and generates concise summaries using modern NLP models like BART or T5. Designed for lecture notes, meetings, and personal recordings, Summy operates fully offline to ensure user data privacy.

## Prerequisites
Please ensure these are present

- Node.js (v16 or higher)
- Python (v3.8 or higher)
- Rust (for Tauri)
- Xcode (full version) (for Tauri)
- cmake

## Backend Setup

1. Navigate to the server directory:
   ```bash
   cd server
   ```

2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the backend server:
   ```bash
   python app.py
   ```
   This will start the Flask server.

## Frontend Setup

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install Node.js dependencies:
   ```bash
   npm install
   ```

3. Run the development server:
   ```bash
   npm run tauri dev
   ```
   This will start the Tauri application with the React frontend.

## Development

- The backend server runs on `http://localhost:8178`
- The frontend development server will open automatically through Tauri
- Any changes to the frontend code will automatically reload the application
- Backend changes require a server restart

## Project Structure

- `frontend/` - Contains the Tauri + React frontend application
- `server/` - Contains the Python Flask backend
  - `app.py` - Main server file
  - `requirements.txt` - Python dependencies
  - `setup.sh` - Server setup and run script

## Troubleshooting

If you encounter any issues:

1. Make sure all prerequisites are installed
2. Ensure the backend server is running before starting the frontend
3. Check that ports 8178 is not being used by other applications
4. For backend issues, check the Python virtual environment is activated
