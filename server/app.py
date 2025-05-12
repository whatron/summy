from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import pipeline
import numpy as np
import os
from datetime import datetime
import glob
import time
from scipy import signal

app = Flask(__name__)
CORS(app)

# Constants
WHISPER_SAMPLE_RATE = 16000  # 16kHz sample rate
AUDIO_DIR = "audio"
MAX_AUDIO_AGE_HOURS = 24  # Keep audio files for 24 hours

# Create audio directory if it doesn't exist
if not os.path.exists(AUDIO_DIR):
    os.makedirs(AUDIO_DIR)

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

def save_audio_as_wav(audio_data, sample_rate, filename):
    """Save audio data as a WAV file with proper headers"""
    import wave
    import struct
    
    # Convert float32 samples to int16
    audio_data = (audio_data * 32767).astype(np.int16)
    
    # Create WAV file
    with wave.open(filename, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data.tobytes())

# def resample_audio(audio_data, original_rate, target_rate):
#     """Resample audio data to target sample rate"""
#     # Calculate number of samples in resampled audio
#     num_samples = int(len(audio_data) * target_rate / original_rate)
    
#     # Print debug info
#     print(f"Resampling: {len(audio_data)} samples at {original_rate}Hz to {num_samples} samples at {target_rate}Hz")
#     print(f"Original duration: {len(audio_data)/original_rate:.3f}s")
#     print(f"Target duration: {num_samples/target_rate:.3f}s")
    
#     # Resample using scipy's resample function
#     resampled = signal.resample(audio_data, num_samples)
    
#     # Verify the resampling
#     if abs(len(resampled)/target_rate - len(audio_data)/original_rate) > 0.001:
#         print(f"Warning: Duration mismatch after resampling! Original: {len(audio_data)/original_rate:.3f}s, Resampled: {len(resampled)/target_rate:.3f}s")
    
#     return resampled

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
            
            # Read the raw audio data
            raw_data = audio_file.read()
            
            # Convert raw bytes to numpy array of float32 samples
            audio_array = np.frombuffer(raw_data, dtype=np.float32)
            
            # Normalize the audio if needed
            if np.max(np.abs(audio_array)) > 1.0:
                audio_array = audio_array / np.max(np.abs(audio_array))
            
            
            # Save as WAV with proper headers
            save_audio_as_wav(audio_array, WHISPER_SAMPLE_RATE, filepath)
            
            print(f"Saved audio to: {filepath}")
            print(f"Audio stats - Min: {np.min(audio_array):.3f}, Max: {np.max(audio_array):.3f}, Mean: {np.mean(audio_array):.3f}")
            
            # Debug: Check file size and content
            file_size = os.path.getsize(filepath)
            print(f"File size: {file_size} bytes")
            
            # Calculate chunk duration in milliseconds
            num_samples = len(audio_array)
            chunk_duration_ms = (num_samples / WHISPER_SAMPLE_RATE) * 1000
            print(f"Final audio duration: {chunk_duration_ms/1000:.3f}s")
            
            # Initialize the pipeline with better chunk handling
            pipe = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-large-v3",
                torch_dtype=torch.float16,
                device="mps:0" if torch.backends.mps.is_available() else "cpu",
                chunk_length_s=30,
                stride_length_s=5,  # Add stride to prevent overlap
                batch_size=16,
                return_timestamps=True
            )

            # Process the audio with better chunk handling
            outputs = pipe(
                audio_array,
                chunk_length_s=30,
                stride_length_s=5,  # Add stride to prevent overlap
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
                    # Ensure t1 is a valid float, use chunk_duration_ms if None
                    t0 = chunk['timestamp'][0] if isinstance(chunk['timestamp'], tuple) else 0.0
                    t1 = chunk['timestamp'][1] if isinstance(chunk['timestamp'], tuple) and chunk['timestamp'][1] is not None else chunk_duration_ms / 1000.0
                    
                    # Only add non-empty segments
                    if chunk['text'].strip():
                        segments.append({
                            "text": chunk['text'].strip(),
                            "t0": t0,
                            "t1": t1
                        })
            else:
                # Handle single segment output
                text = outputs['text'] if isinstance(outputs, dict) else str(outputs)
                if text.strip():
                    segments.append({
                        "text": text.strip(),
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

@app.route('/process_wav', methods=['POST'])
def process_wav():
    """Process a single WAV file and return the transcription"""
    # Get the WAV file from the request
    wav_file = request.files.get('audio')
    
    if wav_file:
        try:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"audio_{timestamp}.wav"
            filepath = os.path.join(AUDIO_DIR, filename)
            
            # Save the WAV file
            wav_file.save(filepath)
            print(f"Saved WAV file to: {filepath}")
            
            # Read the WAV file
            import wave
            with wave.open(filepath, 'rb') as wav:
                # Get WAV file parameters
                sample_rate = wav.getframerate()
                channels = wav.getnchannels()
                frames = wav.getnframes()
                duration = frames / float(sample_rate)
                
                print(f"WAV file info - Sample rate: {sample_rate}Hz, Channels: {channels}, Duration: {duration:.3f}s")
                
                # Read audio data
                audio_data = wav.readframes(frames)
                audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Convert to mono if needed
                if channels > 1:
                    audio_array = audio_array.reshape(-1, channels)
                    audio_array = np.mean(audio_array, axis=1)
                
                print(f"Audio stats - Min: {np.min(audio_array):.3f}, Max: {np.max(audio_array):.3f}, Mean: {np.mean(audio_array):.3f}")
                
                # Initialize the pipeline
                pipe = pipeline(
                    "automatic-speech-recognition",
                    model="openai/whisper-large-v3",
                    torch_dtype=torch.float16,
                    device="mps:0",
                    chunk_length_s=30,
                    stride_length_s=5,
                    batch_size=16,
                    return_timestamps=True
                )
                
                # Process the audio
                outputs = pipe(
                    audio_array,
                    chunk_length_s=30,
                    stride_length_s=5,
                    batch_size=16,
                    return_timestamps=True
                )
                
                print("Transcription output:", outputs)
                
                # Format the response
                segments = []
                if isinstance(outputs, dict) and 'chunks' in outputs:
                    # Handle chunked output
                    for chunk in outputs['chunks']:
                        # Ensure t1 is a valid float, use duration if None
                        t0 = chunk['timestamp'][0] if isinstance(chunk['timestamp'], tuple) else 0.0
                        t1 = chunk['timestamp'][1] if isinstance(chunk['timestamp'], tuple) and chunk['timestamp'][1] is not None else duration
                        
                        # Only add non-empty segments
                        if chunk['text'].strip():
                            segments.append({
                                "text": chunk['text'].strip(),
                                "t0": t0,
                                "t1": t1
                            })
                else:
                    # Handle single segment output
                    text = outputs['text'] if isinstance(outputs, dict) else str(outputs)
                    if text.strip():
                        segments.append({
                            "text": text.strip(),
                            "t0": 0.0,
                            "t1": duration
                        })
                
                # Clean up the file
                cleanup_old_audio()
                
                # Return the transcription
                return jsonify({
                    "segments": segments,
                    "duration": duration,
                    "sample_rate": sample_rate,
                    "channels": channels,
                    "audio_file": filename
                })
                
        except Exception as e:
            print(f"Error processing WAV file: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return jsonify({"error": f"Failed to process WAV file: {str(e)}"}), 500
    
    return jsonify({"error": "No WAV file received"}), 400

@app.route('/summarize', methods=['POST'])
def summarize():
    """Generate a summary from the transcript using T5-small"""
    try:
        data = request.get_json()
        transcript = data.get('transcript', '')
        
        if not transcript:
            return jsonify({"error": "No transcript provided"}), 400

        # Initialize T5 pipeline
        summarizer = pipeline(
            "summarization",
            model="t5-small",
            device="mps:0" if torch.backends.mps.is_available() else "cpu"
        )

        # Split transcript into chunks if it's too long (T5 has a max input length)
        max_chunk_length = 512
        chunks = [transcript[i:i + max_chunk_length] for i in range(0, len(transcript), max_chunk_length)]
        
        # Process each chunk
        summaries = []
        for chunk in chunks:
            # Calculate appropriate max_length based on input length
            input_length = len(chunk.split())
            max_length = min(max(input_length // 2, 30), 100)  # Between 30 and 100 tokens
            min_length = max(input_length // 4, 20)  # Between 20 and 50% of input
            
            # Generate summary for the chunk
            summary = summarizer(chunk, 
                               max_length=max_length, 
                               min_length=min_length, 
                               do_sample=False)
            summaries.append(summary[0]['summary_text'])

        # Combine summaries
        combined_summary = " ".join(summaries)

        # Extract key information using T5 for specific aspects
        def extract_aspect(text, aspect_prompt):
            input_length = len(text.split())
            max_length = min(max(input_length // 3, 20), 50)  # Shorter for aspect extraction
            min_length = max(input_length // 6, 10)  # Even shorter for aspect extraction
            
            prompt = f"{aspect_prompt}: {text}"
            result = summarizer(prompt, 
                              max_length=max_length, 
                              min_length=min_length, 
                              do_sample=False)
            return result[0]['summary_text'].split(", ")

        # Generate structured summary
        summary = {
            "summary": {
                "main_concepts": extract_aspect(combined_summary, "Extract main concepts"),
                "key_definitions": extract_aspect(combined_summary, "Extract key definitions"),
                "important_formulas": extract_aspect(combined_summary, "Extract important formulas"),
                "examples": extract_aspect(combined_summary, "Extract examples"),
                "learning_objectives": extract_aspect(combined_summary, "Extract learning objectives"),
                "prerequisites": extract_aspect(combined_summary, "Extract prerequisites"),
                "sentiment": "informative",  # Could be enhanced with sentiment analysis
                "confidence_score": 0.85,  # Could be enhanced with confidence scoring
                "difficulty_level": "intermediate"  # Could be enhanced with difficulty analysis
            },
            "transcript_id": f"lecture_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "processing_time": 1.5,
            "lecture_metadata": {
                "subject": "General",  # Could be enhanced with subject detection
                "topic": "Lecture",    # Could be enhanced with topic detection
                "level": "General",    # Could be enhanced with level detection
                "estimated_duration": "60 minutes"  # Could be enhanced with duration estimation
            }
        }
        
        return jsonify(summary)
        
    except Exception as e:
        print(f"Error generating summary: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": f"Failed to generate summary: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8178, debug=True) 