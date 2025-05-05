#!/bin/bash

# Start the server in the background
echo "Starting server..."
cd server

# Setup the server
source setup.sh

# Start the server
python app.py &
SERVER_PID=$!
cd ..

# Start the frontend
echo "Starting frontend..."
cd frontend
npm run tauri dev

# Cleanup function to kill the server when the script is terminated
cleanup() {
    echo "Stopping server..."
    kill $SERVER_PID
    exit 0
}

# Set up trap to catch termination signals
trap cleanup SIGINT SIGTERM