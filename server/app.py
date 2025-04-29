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
            
            # Convert to float32 and normalize
            audio_array = audio_array.astype(np.float32) / 32768.0
            
            # Initialize the pipeline
            pipe = pipeline("automatic-speech-recognition",
                    "openai/whisper-medium",
                    torch_dtype=torch.float16,
                    device="mps:0")

            # Process the audio
            outputs = pipe(audio_array,
                   chunk_length_s=30,
                   batch_size=16)
            
            # Return the transcription in the expected format
            return jsonify({
                "segments": [
                    {
                        "text": str(outputs),
                        "t0": 0.0,
                        "t1": 1.0
                    }
                ],
                "buffer_size_ms": 1000
            })
        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            return jsonify({"error": f"Failed to process audio: {str(e)}"}), 500
    
    return jsonify({"error": "No audio file received"}), 400

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8178, debug=True) 