use serde::{Deserialize, Serialize};
use std::fs;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::time::Duration;

// Declare audio module
pub mod audio;
pub mod ollama;

use audio::{default_input_device, default_output_device, AudioStream};
use log::{debug as log_debug, error as log_error, info as log_info};
use reqwest::multipart::{Form, Part};
use tauri::{AppHandle, Emitter, Runtime};

static RECORDING_FLAG: AtomicBool = AtomicBool::new(false);
static mut MIC_BUFFER: Option<Arc<Mutex<Vec<f32>>>> = None;
static mut SYSTEM_BUFFER: Option<Arc<Mutex<Vec<f32>>>> = None;
static mut MIC_STREAM: Option<Arc<AudioStream>> = None;
static mut SYSTEM_STREAM: Option<Arc<AudioStream>> = None;
static mut IS_RUNNING: Option<Arc<AtomicBool>> = None;
static mut RECORDING_START_TIME: Option<std::time::Instant> = None;

// Audio configuration constants
const CHUNK_DURATION_MS: u32 = 30000; // 30 seconds per chunk for better sentence processing
const WHISPER_SAMPLE_RATE: u32 = 16000; // Whisper's required sample rate
const SENTENCE_TIMEOUT_MS: u64 = 1000; // Emit incomplete sentence after 1 second of silence
const MIN_CHUNK_DURATION_MS: u32 = 2000; // Minimum duration before sending chunk
const MIN_RECORDING_DURATION_MS: u64 = 2000; // 2 seconds minimum

#[derive(Debug, Deserialize)]
struct RecordingArgs {
    save_path: String,
}

#[derive(Debug, Serialize, Clone)]
struct TranscriptUpdate {
    text: String,
    timestamp: String,
    source: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct TranscriptSegment {
    text: String,
    t0: f32,
    t1: f32,
}

#[derive(Debug, Serialize, Deserialize)]
struct TranscriptResponse {
    segments: Vec<TranscriptSegment>,
}

// Helper struct to accumulate transcript segments
#[derive(Debug)]
struct TranscriptAccumulator {
    current_sentence: String,
    sentence_start_time: f32,
    last_update_time: std::time::Instant,
    last_segment_hash: u64,
}

impl TranscriptAccumulator {
    fn new() -> Self {
        Self {
            current_sentence: String::new(),
            sentence_start_time: 0.0,
            last_update_time: std::time::Instant::now(),
            last_segment_hash: 0,
        }
    }

    fn add_segment(&mut self, segment: &TranscriptSegment) -> Option<TranscriptUpdate> {
        log_info!("Processing new transcript segment: {:?}", segment);

        // Update the last update time
        self.last_update_time = std::time::Instant::now();

        // Clean up the text (remove [BLANK_AUDIO], [AUDIO OUT] and trim)
        let clean_text = segment
            .text
            .replace("[BLANK_AUDIO]", "")
            .replace("[AUDIO OUT]", "")
            .trim()
            .to_string();

        if !clean_text.is_empty() {
            log_info!("Clean transcript text: {}", clean_text);
        }

        // Skip empty segments or very short segments (less than 1 second)
        if clean_text.is_empty() || (segment.t1 - segment.t0) < 1.0 {
            return None;
        }

        // Calculate hash of this segment to detect duplicates
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        segment.text.hash(&mut hasher);
        segment.t0.to_bits().hash(&mut hasher);
        segment.t1.to_bits().hash(&mut hasher);
        let segment_hash = hasher.finish();

        // Skip if this is a duplicate segment
        if segment_hash == self.last_segment_hash {
            return None;
        }
        self.last_segment_hash = segment_hash;

        // If this is the start of a new sentence, store the start time
        if self.current_sentence.is_empty() {
            self.sentence_start_time = segment.t0;
        }

        // Add the new text with proper spacing
        if !self.current_sentence.is_empty() && !self.current_sentence.ends_with(' ') {
            self.current_sentence.push(' ');
        }
        self.current_sentence.push_str(&clean_text);

        // Check if we have a complete sentence
        if clean_text.ends_with('.') || clean_text.ends_with('?') || clean_text.ends_with('!') {
            let sentence = std::mem::take(&mut self.current_sentence);
            let update = TranscriptUpdate {
                text: sentence.trim().to_string(),
                timestamp: format!("{:.1} - {:.1}", self.sentence_start_time, segment.t1),
                source: "Mixed Audio".to_string(),
            };
            log_info!("Generated transcript update: {:?}", update);
            Some(update)
        } else {
            None
        }
    }

    fn check_timeout(&mut self) -> Option<TranscriptUpdate> {
        if !self.current_sentence.is_empty()
            && self.last_update_time.elapsed() > Duration::from_millis(SENTENCE_TIMEOUT_MS)
        {
            let sentence = std::mem::take(&mut self.current_sentence);
            let current_time = self.sentence_start_time + (SENTENCE_TIMEOUT_MS as f32 / 1000.0);
            let update = TranscriptUpdate {
                text: sentence.trim().to_string(),
                timestamp: format!("{:.1} - {:.1}", self.sentence_start_time, current_time),
                source: "Mixed Audio".to_string(),
            };
            Some(update)
        } else {
            None
        }
    }
}

async fn send_audio_chunk(
    chunk: Vec<f32>,
    client: &reqwest::Client,
) -> Result<TranscriptResponse, String> {
    log_debug!("Preparing to send audio chunk of size: {}", chunk.len());
    
    // Calculate expected duration
    let expected_duration = chunk.len() as f32 / WHISPER_SAMPLE_RATE as f32;
    log_debug!("Expected chunk duration: {:.3}s ({} samples at {}Hz)", 
        expected_duration, chunk.len(), WHISPER_SAMPLE_RATE);

    // Convert f32 samples to bytes, preserving the exact values
    let bytes: Vec<u8> = chunk
        .iter()
        .flat_map(|&sample| sample.to_le_bytes().to_vec())
        .collect();

    log_debug!("Converted {} samples to {} bytes", chunk.len(), bytes.len());

    // Retry configuration
    let max_retries = 3;
    let mut retry_count = 0;
    let mut last_error = String::new();

    while retry_count <= max_retries {
        if retry_count > 0 {
            // Exponential backoff: wait 2^retry_count * 100ms
            let delay = Duration::from_millis(100 * (2_u64.pow(retry_count as u32)));
            log::info!(
                "Retry attempt {} of {}. Waiting {:?} before retry...",
                retry_count,
                max_retries,
                delay
            );
            tokio::time::sleep(delay).await;
        }

        // Create fresh multipart form for each attempt since Form can't be reused
        let part = Part::bytes(bytes.clone())
            .file_name("audio.raw")
            .mime_str("audio/x-raw")
            .unwrap();
        let form = Form::new().part("audio", part);

        match client
            .post("http://127.0.0.1:8178/stream")
            .multipart(form)
            .send()
            .await
        {
            Ok(response) => match response.json::<TranscriptResponse>().await {
                Ok(transcript) => return Ok(transcript),
                Err(e) => {
                    last_error = e.to_string();
                    log::error!("Failed to parse response: {}", last_error);
                }
            },
            Err(e) => {
                last_error = e.to_string();
                log::error!("Request failed: {}", last_error);
            }
        }

        retry_count += 1;
    }

    Err(format!(
        "Failed to send audio chunk after {} retries. Last error: {}",
        max_retries, last_error
    ))
}

#[tauri::command]
async fn start_recording<R: Runtime>(app: AppHandle<R>) -> Result<(), String> {
    log_info!("Attempting to start recording...");

    if is_recording() {
        log_error!("Recording already in progress");
        return Err("Recording already in progress".to_string());
    }

    // Initialize recording flag and buffers
    RECORDING_FLAG.store(true, Ordering::SeqCst);
    log_info!("Recording flag set to true");

    // Store recording start time
    unsafe {
        RECORDING_START_TIME = Some(std::time::Instant::now());
    }

    // Initialize audio buffers
    unsafe {
        MIC_BUFFER = Some(Arc::new(Mutex::new(Vec::new())));
        SYSTEM_BUFFER = Some(Arc::new(Mutex::new(Vec::new())));
        log_info!("Initialized audio buffers");
    }

    // Get default devices
    let mic_device = Arc::new(default_input_device().map_err(|e| {
        log_error!("Failed to get default input device: {}", e);
        e.to_string()
    })?);

    let system_device = Arc::new(default_output_device().map_err(|e| {
        log_error!("Failed to get default output device: {}", e);
        e.to_string()
    })?);

    // Create audio streams
    let is_running = Arc::new(AtomicBool::new(true));

    // Create microphone stream
    let mic_stream = AudioStream::from_device(mic_device.clone(), is_running.clone())
        .await
        .map_err(|e| {
            log_error!("Failed to create microphone stream: {}", e);
            e.to_string()
        })?;
    let mic_stream = Arc::new(mic_stream);

    // Create system audio stream
    let system_stream = AudioStream::from_device(system_device.clone(), is_running.clone())
        .await
        .map_err(|e| {
            log_error!("Failed to create system stream: {}", e);
            e.to_string()
        })?;
    let system_stream = Arc::new(system_stream);

    unsafe {
        MIC_STREAM = Some(mic_stream.clone());
        SYSTEM_STREAM = Some(system_stream.clone());
        IS_RUNNING = Some(is_running.clone());
    }

    // Create HTTP client for transcription
    let client = reqwest::Client::new();

    // Start transcription task
    let app_handle = app.clone();

    // Create audio receivers
    let mic_receiver = mic_stream.subscribe().await;
    let mut mic_receiver_clone = mic_receiver.resubscribe();
    let mut system_receiver = system_stream.subscribe().await;

    // Create debug directory for chunks in temp
    let temp_dir = std::env::temp_dir();
    log_info!("System temp directory: {:?}", temp_dir);
    let debug_dir = temp_dir.join("meeting_minutes_debug");
    log_info!("Full debug directory path: {:?}", debug_dir);

    // Create directory and check if it exists
    fs::create_dir_all(&debug_dir).map_err(|e| {
        log_error!("Failed to create debug directory: {}", e);
        e.to_string()
    })?;

    if debug_dir.exists() {
        log_info!("Debug directory successfully created and exists");
    } else {
        log_error!("Failed to create debug directory - path does not exist after creation");
    }

    let chunk_counter = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let chunk_counter_clone = chunk_counter.clone();

    // Create transcript accumulator
    let mut accumulator = TranscriptAccumulator::new();

    let device_config = mic_stream.device_config.clone();
    let _device_name = mic_stream.device.to_string();
    let sample_rate = device_config.sample_rate().0;
    let channels = device_config.channels();

    tokio::spawn(async move {
        let chunk_samples =
            (WHISPER_SAMPLE_RATE as f32 * (CHUNK_DURATION_MS as f32 / 1000.0)) as usize;
        let min_samples =
            (WHISPER_SAMPLE_RATE as f32 * (MIN_CHUNK_DURATION_MS as f32 / 1000.0)) as usize;
        let mut current_chunk: Vec<f32> = Vec::with_capacity(chunk_samples);
        let mut last_chunk_time = std::time::Instant::now();

        log_info!("Mic config: {} Hz, {} channels", sample_rate, channels);

        while is_running.load(Ordering::SeqCst) {
            // Check for timeout on current sentence
            if let Some(update) = accumulator.check_timeout() {
                if let Err(e) = app_handle.emit("transcript-update", update) {
                    log_error!("Failed to send timeout transcript update: {}", e);
                }
            }

            // Collect audio samples
            let mut new_samples: Vec<f32> = Vec::new();
            let mut mic_samples: Vec<f32> = Vec::new();
            let mut system_samples: Vec<f32> = Vec::new();

            // Get microphone samples
            let mut got_mic_samples = false;
            while let Ok(chunk) = mic_receiver_clone.try_recv() {
                got_mic_samples = true;
                log_debug!("Received {} mic samples", chunk.len());
                let chunk_clone = chunk.clone();
                mic_samples.extend(chunk);

                // Store in global buffer
                unsafe {
                    if let Some(buffer) = (*&raw const MIC_BUFFER).as_ref() {
                        if let Ok(mut guard) = buffer.lock() {
                            guard.extend(chunk_clone);
                        }
                    }
                }
            }
            // If we didn't get any samples, try to resubscribe to clear any backlog
            if !got_mic_samples {
                log_debug!("No mic samples received, resubscribing to clear channel");
                mic_receiver_clone = mic_stream.subscribe().await;
            }

            // Get system audio samples
            let mut got_system_samples = false;
            while let Ok(chunk) = system_receiver.try_recv() {
                got_system_samples = true;
                log_debug!("Received {} system samples", chunk.len());
                let chunk_clone = chunk.clone();
                system_samples.extend(chunk);

                // Store in global buffer
                unsafe {
                    if let Some(buffer) = (*&raw const SYSTEM_BUFFER).as_ref() {
                        if let Ok(mut guard) = buffer.lock() {
                            guard.extend(chunk_clone);
                        }
                    }
                }
            }
            // If we didn't get any samples, try to resubscribe to clear any backlog
            if !got_system_samples {
                log_debug!("No system samples received, resubscribing to clear channel");
                system_receiver = system_stream.subscribe().await;
            }

            // Mix the audio streams at their original sample rates
            let mic_rate = mic_stream.device_config.sample_rate().0;
            let system_rate = system_stream.device_config.sample_rate().0;
            
            // Calculate relative lengths based on sample rates to maintain timing
            let mic_len = mic_samples.len();
            let system_len = (system_samples.len() as f32 * (mic_rate as f32 / system_rate as f32)) as usize;
            let min_len = mic_len.min(system_len);
            
            let mut mixed_samples = Vec::with_capacity(min_len);
            
            for i in 0..min_len {
                let mic_sample = if i < mic_samples.len() { mic_samples[i] } else { 0.0 };
                let system_idx = (i as f32 * (system_rate as f32 / mic_rate as f32)) as usize;
                let system_sample = if system_idx < system_samples.len() { system_samples[system_idx] } else { 0.0 };
                // Mix with 70% mic and 30% system audio
                mixed_samples.push((mic_sample * 0.7) + (system_sample * 0.3));
            }

            log_debug!("Mixed {} samples at {}Hz", mixed_samples.len(), mic_rate);

            // Normalize the mixed audio
            let normalized_samples = audio::audio_processing::normalize_v2(&mixed_samples);
            log_debug!("Normalized {} samples", normalized_samples.len());

            // Apply spectral subtraction for noise reduction
            let noise_floor = audio::audio_processing::average_noise_spectrum(&normalized_samples);
            let denoised_samples = match audio::audio_processing::spectral_subtraction(&normalized_samples, noise_floor) {
                Ok(denoised) => denoised,
                Err(e) => {
                    log_error!("Failed to apply spectral subtraction: {}", e);
                    normalized_samples // Fallback to normalized samples if denoising fails
                }
            };
            log_debug!("Applied spectral subtraction with noise floor: {}", noise_floor);

            // Resample the mixed audio to 16kHz
            let resampled_samples = if mic_rate != WHISPER_SAMPLE_RATE {
                match audio::audio_processing::resample(&denoised_samples, mic_rate, WHISPER_SAMPLE_RATE) {
                    Ok(resampled) => resampled,
                    Err(e) => {
                        log_error!("Failed to resample mixed audio: {}", e);
                        continue;
                    }
                }
            } else {
                denoised_samples
            };

            // Add samples to current chunk
            for sample in resampled_samples {
                current_chunk.push(sample);
            }

            // Check if we should send the chunk based on size or time
            let should_send = current_chunk.len() >= chunk_samples
                || (current_chunk.len() >= min_samples
                    && last_chunk_time.elapsed()
                        >= Duration::from_millis(CHUNK_DURATION_MS as u64));

            if should_send {
                log_info!("Should send chunk with {} samples", current_chunk.len());
                let chunk_to_send = current_chunk.clone();
                current_chunk.clear();
                last_chunk_time = std::time::Instant::now();

                // Save debug chunks
                let chunk_num = chunk_counter_clone.fetch_add(1, Ordering::SeqCst);
                log_info!("Processing chunk {}", chunk_num);

                // Send chunk for transcription
                match send_audio_chunk(chunk_to_send, &client).await {
                    Ok(response) => {
                        // Process transcription response
                        for segment in response.segments {
                            if let Some(update) = accumulator.add_segment(&segment) {
                                if let Err(e) = app_handle.emit("transcript-update", update) {
                                    log_error!("Failed to emit transcript update: {}", e);
                                }
                            }
                        }
                    }
                    Err(e) => {
                        log_error!("Failed to send audio chunk: {}", e);
                    }
                }
            }

            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        // Emit any remaining transcript when recording stops
        if let Some(update) = accumulator.check_timeout() {
            if let Err(e) = app_handle.emit("transcript-update", update) {
                log_error!("Failed to send final transcript update: {}", e);
            }
        }

        log_info!("Transcription task ended");
    });

    Ok(())
}

#[tauri::command]
async fn stop_recording(args: RecordingArgs) -> Result<(), String> {
    log_info!("Attempting to stop recording...");

    // Only check recording state if we haven't already started stopping
    if !RECORDING_FLAG.load(Ordering::SeqCst) {
        log_info!("Recording is already stopped");
        return Ok(());
    }

    // Check minimum recording duration
    let elapsed_ms = unsafe {
        RECORDING_START_TIME
            .map(|start| start.elapsed().as_millis() as u64)
            .unwrap_or(0)
    };

    if elapsed_ms < MIN_RECORDING_DURATION_MS {
        let remaining = MIN_RECORDING_DURATION_MS - elapsed_ms;
        log_info!(
            "Waiting for minimum recording duration ({} ms remaining)...",
            remaining
        );
        tokio::time::sleep(Duration::from_millis(remaining)).await;
    }

    // First set the recording flag to false to prevent new data from being processed
    RECORDING_FLAG.store(false, Ordering::SeqCst);
    log_info!("Recording flag set to false");

    unsafe {
        // Stop the running flag for audio streams first
        if let Some(is_running) = (*&raw const IS_RUNNING).as_ref() {
            // Set running flag to false first to stop the tokio task
            is_running.store(false, Ordering::SeqCst);
            log_info!("Set recording flag to false, waiting for streams to stop...");

            // Give the tokio task time to finish and release its references
            tokio::time::sleep(Duration::from_millis(100)).await;

            // Stop mic stream if it exists
            if let Some(mic_stream) = (*&raw const MIC_STREAM).as_ref() {
                log_info!("Stopping microphone stream...");
                if let Err(e) = mic_stream.stop().await {
                    log_error!("Error stopping mic stream: {}", e);
                } else {
                    log_info!("Microphone stream stopped successfully");
                }
            }

            // Stop system stream if it exists
            if let Some(system_stream) = (*&raw const SYSTEM_STREAM).as_ref() {
                log_info!("Stopping system stream...");
                if let Err(e) = system_stream.stop().await {
                    log_error!("Error stopping system stream: {}", e);
                } else {
                    log_info!("System stream stopped successfully");
                }
            }

            // Clear the stream references
            MIC_STREAM = None;
            SYSTEM_STREAM = None;
            IS_RUNNING = None;

            // Give streams time to fully clean up
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    // Get final buffers
    let mic_data = unsafe {
        if let Some(buffer) = (*&raw const MIC_BUFFER).as_ref() {
            if let Ok(guard) = buffer.lock() {
                guard.clone()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        }
    };

    let system_data = unsafe {
        if let Some(buffer) = (*&raw const SYSTEM_BUFFER).as_ref() {
            if let Ok(guard) = buffer.lock() {
                guard.clone()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        }
    };

    // Mix the audio and convert to 16-bit PCM
    let max_len = mic_data.len().max(system_data.len());
    let mut mixed_data = Vec::with_capacity(max_len);

    for i in 0..max_len {
        let mic_sample = if i < mic_data.len() { mic_data[i] } else { 0.0 };
        let system_sample = if i < system_data.len() {
            system_data[i]
        } else {
            0.0
        };
        mixed_data.push((mic_sample + system_sample) * 0.5);
    }

    if mixed_data.is_empty() {
        log_error!("No audio data captured");
        return Err("No audio data captured".to_string());
    }

    log_info!("Mixed {} audio samples", mixed_data.len());

    // Resample the audio to 16kHz for Whisper compatibility
    let original_sample_rate = 48000; // Assuming original sample rate is 48kHz
    if original_sample_rate != WHISPER_SAMPLE_RATE {
        log_info!(
            "Resampling audio from {} Hz to {} Hz for Whisper compatibility",
            original_sample_rate,
            WHISPER_SAMPLE_RATE
        );
        mixed_data = resample_audio(&mixed_data, original_sample_rate, WHISPER_SAMPLE_RATE);
        log_info!("Resampled to {} samples", mixed_data.len());
    }

    // Convert to 16-bit PCM samples
    let mut bytes = Vec::with_capacity(mixed_data.len() * 2);
    for &sample in mixed_data.iter() {
        let value = (sample.max(-1.0).min(1.0) * 32767.0) as i16;
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    log_info!("Converted to {} bytes of PCM data", bytes.len());

    // Create WAV header
    let data_size = bytes.len() as u32;
    let file_size = 36 + data_size;
    let sample_rate = WHISPER_SAMPLE_RATE; // Use Whisper's required sample rate (16000 Hz)
    let channels = 1u16; // Mono
    let bits_per_sample = 16u16;
    let block_align = channels * (bits_per_sample / 8);
    let byte_rate = sample_rate * block_align as u32;

    let mut wav_file = Vec::with_capacity(44 + bytes.len());

    // RIFF header
    wav_file.extend_from_slice(b"RIFF");
    wav_file.extend_from_slice(&file_size.to_le_bytes());
    wav_file.extend_from_slice(b"WAVE");

    // fmt chunk
    wav_file.extend_from_slice(b"fmt ");
    wav_file.extend_from_slice(&16u32.to_le_bytes()); // fmt chunk size
    wav_file.extend_from_slice(&1u16.to_le_bytes()); // audio format (PCM)
    wav_file.extend_from_slice(&channels.to_le_bytes()); // num channels
    wav_file.extend_from_slice(&sample_rate.to_le_bytes()); // sample rate
    wav_file.extend_from_slice(&byte_rate.to_le_bytes()); // byte rate
    wav_file.extend_from_slice(&block_align.to_le_bytes()); // block align
    wav_file.extend_from_slice(&bits_per_sample.to_le_bytes()); // bits per sample

    // data chunk
    wav_file.extend_from_slice(b"data");
    wav_file.extend_from_slice(&data_size.to_le_bytes());
    wav_file.extend_from_slice(&bytes);

    log_info!("Created WAV file with {} bytes total", wav_file.len());

    // Create the save directory if it doesn't exist
    if let Some(parent) = std::path::Path::new(&args.save_path).parent() {
        if !parent.exists() {
            log_info!("Creating directory: {:?}", parent);
            if let Err(e) = std::fs::create_dir_all(parent) {
                let err_msg = format!("Failed to create save directory: {}", e);
                log_error!("{}", err_msg);
                return Err(err_msg);
            }
        }
    }

    // Save the recording
    log_info!("Saving recording to: {}", args.save_path);
    match fs::write(&args.save_path, wav_file) {
        Ok(_) => log_info!("Successfully saved recording"),
        Err(e) => {
            let err_msg = format!("Failed to save recording: {}", e);
            log_error!("{}", err_msg);
            return Err(err_msg);
        }
    }

    // Clean up
    unsafe {
        MIC_BUFFER = None;
        SYSTEM_BUFFER = None;
        MIC_STREAM = None;
        SYSTEM_STREAM = None;
        IS_RUNNING = None;
        RECORDING_START_TIME = None;
    }

    Ok(())
}

#[tauri::command]
fn is_recording() -> bool {
    RECORDING_FLAG.load(Ordering::SeqCst)
}

#[tauri::command]
fn read_audio_file(file_path: String) -> Result<Vec<u8>, String> {
    match std::fs::read(&file_path) {
        Ok(data) => Ok(data),
        Err(e) => Err(format!("Failed to read audio file: {}", e)),
    }
}

#[tauri::command]
async fn save_transcript(file_path: String, content: String) -> Result<(), String> {
    log::info!("Saving transcript to: {}", file_path);

    // Ensure parent directory exists
    if let Some(parent) = std::path::Path::new(&file_path).parent() {
        if !parent.exists() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create directory: {}", e))?;
        }
    }

    // Write content to file
    std::fs::write(&file_path, content)
        .map_err(|e| format!("Failed to write transcript: {}", e))?;

    log::info!("Transcript saved successfully");
    Ok(())
}

#[tauri::command]
async fn get_file_stats(path: String) -> Result<serde_json::Value, String> {
    use std::fs;
    match fs::metadata(&path) {
        Ok(metadata) => Ok(serde_json::json!({
            "exists": true,
            "size": metadata.len(),
            "created": metadata.created().ok().map(|t| t.duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()),
            "modified": metadata.modified().ok().map(|t| t.duration_since(std::time::UNIX_EPOCH).unwrap().as_secs()),
        })),
        Err(e) => Ok(serde_json::json!({
            "exists": false,
            "error": e.to_string()
        })),
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    log::set_max_level(log::LevelFilter::Info);

    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .setup(|_app| {
            log::info!("Application setup complete");

            // Trigger microphone permission request on startup
            if let Err(e) = audio::core::trigger_audio_permission() {
                log::error!("Failed to trigger audio permission: {}", e);
            }

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            start_recording,
            stop_recording,
            is_recording,
            read_audio_file,
            save_transcript,
            get_file_stats,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

// Helper function to resample audio
fn resample_audio(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return samples.to_vec();
    }

    match audio::audio_processing::resample(samples, from_rate, to_rate) {
        Ok(resampled) => resampled,
        Err(e) => {
            log_error!("Failed to resample audio: {}", e);
            // Return original samples if resampling fails
            samples.to_vec()
        }
    }
}
