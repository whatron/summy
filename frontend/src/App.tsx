import { useEffect, useState, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { RecordingControls } from "./components/RecordingControls";

function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [barHeights, setBarHeights] = useState(['58%', '76%', '58%']);
  const [error, setError] = useState<string>('');
  const [transcript, setTranscript] = useState<string>('');
  const transcriptBoxRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (isRecording) {
      const interval = setInterval(() => {
        setBarHeights(prev => {
          const newHeights = [...prev];
          newHeights[0] = Math.random() * 20 + 10 + 'px';
          newHeights[1] = Math.random() * 20 + 10 + 'px';
          newHeights[2] = Math.random() * 20 + 10 + 'px';
          return newHeights;
        });
      }, 300);

      return () => clearInterval(interval);
    }
  }, [isRecording]);

  // Auto-scroll to bottom when transcript updates
  useEffect(() => {
    if (transcriptBoxRef.current) {
      transcriptBoxRef.current.scrollTop = transcriptBoxRef.current.scrollHeight;
    }
  }, [transcript]);

  const handleRecordingStart = async () => {
    try {
      console.log('Starting recording...');
      setTranscript(''); // Clear previous transcript
      setIsLoading(true); // Set loading state
      
      // First check if we're already recording
      const isCurrentlyRecording = await invoke('is_recording');
      if (isCurrentlyRecording) {
        console.log('Already recording, stopping first...');
        await handleRecordingStop();
      }

      // Start new recording
      await invoke('start_recording');
      console.log('Recording started successfully');
      setIsRecording(true);
    } catch (error) {
      console.error('Failed to start recording:', error);
      alert('Failed to start recording. Check console for details.');
      setIsRecording(false); // Reset state on error
    } finally {
      setIsLoading(false); // Clear loading state
    }
  };
  
  const handleRecordingStop = async () => {
    try {
      console.log('Stopping recording...');
      const { appDataDir } = await import('@tauri-apps/api/path');
      
      const dataDir = await appDataDir();
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const audioPath = `${dataDir}/audio/recording-${timestamp}.wav`;

      // Stop recording and save audio
      await invoke('stop_recording', { 
        args: { 
          save_path: audioPath
        }
      });
      console.log('Recording stopped successfully');
      setIsRecording(false);

      // Debug: Check if file exists and its size
      const fileStats = await invoke('get_file_stats', { path: audioPath });
      console.log('Audio file stats:', fileStats);

      // Send audio to server
      const formData = new FormData();
      const audioFile = await fetch(audioPath).then(r => r.blob());
      console.log('Audio blob:', {
        size: audioFile.size,
        type: audioFile.type
      });
      formData.append('audio', audioFile, `recording-${timestamp}.wav`);

      const response = await fetch('http://localhost:8178/process_wav', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}: ${await response.text()}`);
      }

      const data = await response.json();
      if (data.segments && data.segments.length > 0) {
        const transcript = data.segments.map((s: any) => s.text).join(' ');
        replaceTranscript(transcript);
      }
    } catch (error) {
      console.error('Failed to stop recording:', error);
      if (error instanceof Error) {
        console.error('Error details:', {
          message: error.message,
          name: error.name,
          stack: error.stack,
        });
      }
      alert('Failed to stop recording. Check console for details.');
      setIsRecording(false); // Reset state on error
    }
  };

  const handleTranscriptReceived = (newTranscript: string) => {
    setTranscript(prev => {
      // If there's existing text, add a space before the new text
      const separator = prev ? ' ' : '';
      return prev + separator + newTranscript;
    });
  };

  const replaceTranscript = (newTranscript: string) => {
    setTranscript(newTranscript);
  };

  return (
    <main className="container">
      {/* Fixed transcript box */}
      <div className="absolute bottom-32 left-1/2 transform -translate-x-1/2 w-4/5 max-w-4xl">
        <div 
          className="bg-white rounded-lg p-4 h-64 overflow-y-auto border-2 border-gray-300 shadow-[0_4px_12px_rgba(0,0,0,0.15)]" 
          ref={transcriptBoxRef}
        >
          <h2 className="text-lg font-semibold mb-2 text-gray-800 sticky top-0 bg-white pb-2 border-b border-gray-200">Transcript</h2>
          <p className="text-gray-600 whitespace-pre-wrap">{transcript || 'No transcript yet...'}</p>
        </div>
      </div>

      {/* Recording controls */}
      <div className="absolute bottom-16 left-1/2 transform -translate-x-1/2 z-10">
            <div className="bg-white rounded-full shadow-lg flex items-center">
              <RecordingControls
                isRecording={isRecording}
                isLoading={isLoading}
                onRecordingStop={handleRecordingStop}
                onRecordingStart={handleRecordingStart}
                onTranscriptReceived={handleTranscriptReceived}
                barHeights={barHeights}
              />
            </div>
          </div>
    </main>
  );
}

export default App;
