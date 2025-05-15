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
import logging
import sys

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

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
    audio_file = request.files.get('audio')
    
    if audio_file:
        try:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"audio_{timestamp}.wav"
            filepath = os.path.join(AUDIO_DIR, filename)
            
            # Read the raw audio data
            raw_data = audio_file.read()
            
            # First interpret as float32 bytes (4 bytes per sample)
            audio_array = np.frombuffer(raw_data, dtype=np.float32)
            
            # Calculate expected duration based on sample rate and number of samples
            expected_duration = len(audio_array) / WHISPER_SAMPLE_RATE
            print(f"Expected audio duration: {expected_duration:.3f}s ({len(audio_array)} samples at {WHISPER_SAMPLE_RATE}Hz)")
            
            # Debug info
            print(f"Raw audio stats - Min: {np.min(audio_array):.3f}, Max: {np.max(audio_array):.3f}, Mean: {np.mean(audio_array):.3f}")
            print(f"Sample count: {len(audio_array)}")
            
            # Save as WAV for debugging
            save_audio_as_wav(audio_array, WHISPER_SAMPLE_RATE, filepath)
            print(f"Saved debug audio to: {filepath}")
            
            # Initialize the pipeline with better chunk handling
            pipe = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-large-v3",
                torch_dtype=torch.float16,
                device="mps:0" if torch.backends.mps.is_available() else "cpu",
                chunk_length_s=30,
                stride_length_s=5,
                batch_size=16,
                return_timestamps=True
            )

            # Process the audio with explicit sample rate
            outputs = pipe(
                {"array": audio_array, "sampling_rate": WHISPER_SAMPLE_RATE},
                chunk_length_s=30,
                stride_length_s=5,
                batch_size=16,
                return_timestamps=True
            )
            
            print("Transcription output:", outputs)
            print(f"Processed {len(audio_array)} samples ({expected_duration:.3f}s of audio)")
            
            # Format the response according to the frontend's expected structure
            segments = []
            if isinstance(outputs, dict) and 'chunks' in outputs:
                # Handle chunked output
                for chunk in outputs['chunks']:
                    # Ensure t1 is a valid float, use expected_duration if None
                    t0 = chunk['timestamp'][0] if isinstance(chunk['timestamp'], tuple) else 0.0
                    t1 = chunk['timestamp'][1] if isinstance(chunk['timestamp'], tuple) and chunk['timestamp'][1] is not None else expected_duration
                    
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
                        "t1": expected_duration
                    })
            
            # Clean up old audio files
            cleanup_old_audio()
            
            # Return the transcription in the expected format
            return jsonify({
                "segments": segments,
                "duration": expected_duration,
                "audio_file": filename
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
        start_time = time.time()
        data = request.get_json()
        transcript = data.get('transcript', '')
        
        logger.info("Received transcript for summarization")
        logger.debug(f"Transcript length: {len(transcript)} characters")
        
        if not transcript:
            logger.error("No transcript provided")
            return jsonify({"error": "No transcript provided"}), 400

        # Initialize T5 pipeline
        logger.info("Initializing T5 summarization pipeline")
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device="mps:0" if torch.backends.mps.is_available() else "cpu"
        )
        logger.info(f"Using device: {'mps:0' if torch.backends.mps.is_available() else 'cpu'}")

        # Split transcript into chunks if it's too long (T5 has a max input length)
        max_chunk_length = 512
        chunks = [transcript[i:i + max_chunk_length] for i in range(0, len(transcript), max_chunk_length)]
        logger.info(f"Split transcript into {len(chunks)} chunks")
        
        # Process each chunk
        summaries = []
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"Processing chunk {i}/{len(chunks)}")
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
            logger.debug(f"Chunk {i} summary length: {len(summary[0]['summary_text'])} characters")

        # Combine summaries
        combined_summary = " ".join(summaries)
        logger.info(f"Generated combined summary of length: {len(combined_summary)} characters")

        # Calculate actual processing time
        processing_time = time.time() - start_time
        logger.info(f"Total processing time: {processing_time:.2f} seconds")

        # Generate response with actual processing time
        summary = {
            "summary": combined_summary,
            "transcript_id": f"lecture_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "processing_time": round(processing_time, 2),  # Round to 2 decimal places
        }
        
        return jsonify(summary)
    except Exception as e:
        logger.error(f"Error in summarization: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    logger.info("Starting Flask server")
    app.run(host='0.0.0.0', port=8178, debug=True) 