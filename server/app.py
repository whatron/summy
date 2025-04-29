from flask import Flask, request, jsonify
from flask_cors import CORS
import random
import torch
from transformers import pipeline, AutoTokenizer
import os
from pydub import AudioSegment
import numpy as np
import io

app = Flask(__name__)
CORS(app)

# Constants
WHISPER_SAMPLE_RATE = 16000  # 16kHz sample rate

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


@app.route('/stream', methods=['POST'])
def handle_audio():
    # Get the audio file from the request
    audio_file = request.files.get('audio')
    
    if audio_file:
        try:
            # Read the audio file into memory
            audio_bytes = audio_file.read()
            
            # Convert raw bytes to numpy array (assuming 16-bit PCM)
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # Calculate chunk duration in milliseconds
            num_samples = len(audio_array)
            chunk_duration_ms = (num_samples / WHISPER_SAMPLE_RATE) * 1000
            
            # Convert to float32 and normalize
            audio_array = audio_array.astype(np.float32) / 32768.0
            
            # Initialize the pipeline
            pipe = pipeline("automatic-speech-recognition",
                    "openai/whisper-large-v2",
                    torch_dtype=torch.float16,
                    device="mps:0")

            # Process the audio
            outputs = pipe(audio_array,
                   chunk_length_s=30,
                   batch_size=16,
                   return_timestamps=True)  # Enable timestamp return
            
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
            
            # Return the transcription in the expected format
            return jsonify({
                "segments": segments,
                "buffer_size_ms": int(chunk_duration_ms)
            })
        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            return jsonify({"error": f"Failed to process audio: {str(e)}"}), 500
    
    return jsonify({"error": "No audio file received"}), 400

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8178, debug=True) 