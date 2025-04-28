import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import "./App.css";
import { RecordingControls } from "./components/RecordingControls";

function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [barHeights, setBarHeights] = useState(['58%', '76%', '58%']);
  const [error, setError] = useState<string>('');
  const [transcript, setTranscript] = useState<string>('');

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

  const handleRecordingStart = async () => {
    try {
      console.log('Starting recording...');
      setTranscript(''); // Clear previous transcript
      
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
    }
  };
  
  const handleRecordingStop = async () => {
    try {
      console.log('Stopping recording...');
      const { appDataDir } = await import('@tauri-apps/api/path');
      
      const dataDir = await appDataDir();
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const audioPath = `${dataDir}recording-${timestamp}.wav`;

      // Stop recording and save audio
      await invoke('stop_recording', { 
        args: { 
          save_path: audioPath
        }
      });
      console.log('Recording stopped successfully');
      setIsRecording(false);
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

  const handleTranscriptReceived = (transcript: string) => {
    setTranscript(prev => prev + ' ' + transcript);
  };

  return (
    <main className="container">
      {/* Transcription display */}
      {transcript && (
        <div className="absolute top-16 left-1/2 transform -translate-x-1/2 w-3/4 max-w-2xl">
          <div className="bg-white rounded-lg shadow-lg p-4 max-h-96 overflow-y-auto">
            <h2 className="text-lg font-semibold mb-2 text-gray-800">Transcription</h2>
            <p className="text-gray-600 whitespace-pre-wrap">{transcript}</p>
          </div>
        </div>
      )}

      {/* Recording controls */}
      <div className="absolute bottom-16 left-1/2 transform -translate-x-1/2 z-10">
        <div className="bg-white rounded-full shadow-lg flex items-center">
          <RecordingControls
            isRecording={isRecording}
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
