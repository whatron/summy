from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import pipeline
import numpy as np
import os
from datetime import datetime
import glob
import time

app = Flask(__name__)
CORS(app)

# Constants
WHISPER_SAMPLE_RATE = 16000  # 16kHz sample rate
AUDIO_DIR = "audio"
MAX_AUDIO_AGE_HOURS = 24  # Keep audio files for 24 hours

# Create audio directory if it doesn't exist
if not os.path.exists(AUDIO_DIR):
    os.makedirs(AUDIO_DIR)

# List of sample sentences for random transcription
SAMPLE_SENTENCES = [
    "Hello, how are you today?",
    "The weather is beautiful outside.",
    "I'm working on an interesting project.",
    "Can you help me with this task?",
    "Let's meet for coffee tomorrow.",
    "The meeting went well.",
    "I need to finish this report by Friday.",
    "What's your favorite programming language?",
    "The new feature is working as expected.",
    "Let's discuss this in more detail.",
]

def cleanup_old_audio():
    """Remove audio files older than MAX_AUDIO_AGE_HOURS"""
    current_time = time.time()
    for audio_file in glob.glob(os.path.join(AUDIO_DIR, "*.wav")):
        file_age = current_time - os.path.getmtime(audio_file)
        if file_age > MAX_AUDIO_AGE_HOURS * 3600:  # Convert hours to seconds
            try:
                os.remove(audio_file)
                print(f"Removed old audio file: {audio_file}")
            except Exception as e:
                print(f"Error removing file {audio_file}: {str(e)}")

@app.route('/stream', methods=['POST'])
def handle_audio():
    # Get the audio file from the request
    audio_file = request.files.get('audio')
    
    if audio_file:
        try:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"audio_{timestamp}.wav"
            filepath = os.path.join(AUDIO_DIR, filename)
            
            # Save the audio file
            audio_file.save(filepath)
            print(f"Saved audio to: {filepath}")
            
            # Debug: Check file size and content
            file_size = os.path.getsize(filepath)
            print(f"File size: {file_size} bytes")
            
            # Read the raw audio data
            with open(filepath, 'rb') as f:
                raw_data = f.read()
            
            # Convert raw bytes to numpy array of float32 samples
            # Assuming the data is 16-bit PCM (2 bytes per sample)
            audio_array = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Calculate chunk duration in milliseconds
            num_samples = len(audio_array)
            chunk_duration_ms = (num_samples / WHISPER_SAMPLE_RATE) * 1000
            
            # Initialize the pipeline
            pipe = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-medium.en",
                torch_dtype=torch.float16,
                device="mps:0"
            )

            # Process the audio
            outputs = pipe(
                audio_array,
                chunk_length_s=30,
                batch_size=16,
                return_timestamps=True
            )
            
            print("Transcription output:", outputs)
            print(f"Processed {num_samples} samples ({chunk_duration_ms:.2f}ms of audio)")
            
            # Format the response according to the frontend's expected structure
            segments = []
            if isinstance(outputs, dict) and 'chunks' in outputs:
                # Handle chunked output
                for chunk in outputs['chunks']:
                    segments.append({
                        "text": chunk['text'],
                        "t0": chunk['timestamp'][0],
                        "t1": chunk['timestamp'][1]
                    })
            else:
                # Handle single segment output
                segments.append({
                    "text": outputs['text'] if isinstance(outputs, dict) else str(outputs),
                    "t0": 0.0,
                    "t1": chunk_duration_ms / 1000.0
                })
            
            # Clean up old audio files
            cleanup_old_audio()
            
            # Return the transcription in the expected format
            return jsonify({
                "segments": segments,
                "buffer_size_ms": int(chunk_duration_ms),
                "audio_file": filename  # Return the filename to the frontend
            })
        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return jsonify({"error": f"Failed to process audio: {str(e)}"}), 500
    
    return jsonify({"error": "No audio file received"}), 400

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8178, debug=True) 