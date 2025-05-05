use super::ffmpeg::find_ffmpeg_path;
use super::AudioDevice;
use std::path::PathBuf;
use hound::{WavWriter, WavSpec};
use tracing::{debug, error};
use std::sync::Arc;

pub struct AudioInput {
    pub data: Arc<Vec<f32>>,
    pub sample_rate: u32,
    pub channels: u16,
    pub device: Arc<AudioDevice>,
}

pub fn encode_single_audio(
    data: &[u8],
    sample_rate: u32,
    channels: u16,
    output_path: &PathBuf,
) -> anyhow::Result<()> {
    debug!("Starting WAV file encoding");

    // Convert bytes to i16 samples
    let samples: Vec<i16> = data.chunks(2)
        .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
        .collect();

    // Create WAV spec
    let spec = WavSpec {
        channels: channels as u16,
        sample_rate: sample_rate as u32,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    // Create WAV writer
    let mut writer = WavWriter::create(output_path, spec)?;

    // Write samples directly
    for sample in samples {
        writer.write_sample(sample)?;
    }

    writer.finalize()?;
    debug!("WAV file encoding complete");

    Ok(())
}
