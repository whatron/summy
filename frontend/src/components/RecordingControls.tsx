'use client';

import { invoke } from '@tauri-apps/api/core';
import { appDataDir } from '@tauri-apps/api/path';
import { useCallback, useEffect, useState, useRef } from 'react';
import { Play, Pause, Square, Mic, FileText } from 'lucide-react';
import { ProcessRequest, SummaryResponse } from '../types/summary';
import { listen } from '@tauri-apps/api/event';

interface RecordingControlsProps {
  isRecording: boolean;
  isLoading: boolean;
  barHeights: string[];
  onRecordingStop: () => void;
  onRecordingStart: () => void;
  onTranscriptReceived: (transcript: string) => void;
  onSummaryReceived?: (summary: SummaryResponse) => void;
  currentTranscript?: string;
}

export const RecordingControls: React.FC<RecordingControlsProps> = ({
  isRecording,
  isLoading,
  barHeights,
  onRecordingStop,
  onRecordingStart,
  onTranscriptReceived,
  onSummaryReceived,
  currentTranscript,
}) => {
  const [showPlayback, setShowPlayback] = useState(false);
  const [recordingPath, setRecordingPath] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [isStopping, setIsStopping] = useState(false);
  const [stopCountdown, setStopCountdown] = useState(5);
  const countdownInterval = useRef<ReturnType<typeof setInterval> | null>(null);
  const stopTimeoutRef = useRef<{ stop: () => void } | null>(null);
  const MIN_RECORDING_DURATION = 2000; // 2 seconds minimum recording time

  const currentTime = 0;
  const duration = 0;
  const isPlaying = false;
  const progress = 0;

  const formatTime = (time: number) => {
    const minutes = Math.floor(time / 60);
    const seconds = Math.floor(time % 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  };

  useEffect(() => {
    const checkTauri = async () => {
      try {
        const result = await invoke('is_recording');
        console.log('Tauri is initialized and ready, is_recording result:', result);
      } catch (error) {
        console.error('Tauri initialization error:', error);
        alert('Failed to initialize recording. Please check the console for details.');
      }
    };
    checkTauri();
  }, []);

  // Listen for transcript updates
  useEffect(() => {
    if (isRecording) {
      const unlisten = listen('transcript-update', (event: any) => {
        if (event.payload && event.payload.text) {
          onTranscriptReceived(event.payload.text);
        }
      });

      return () => {
        unlisten.then(fn => fn());
      };
    }
  }, [isRecording, onTranscriptReceived]);

  const handleStartRecording = useCallback(async () => {
    if (isStarting) return;
    console.log('Starting recording...');
    setIsStarting(true);
    setShowPlayback(false);
    
    try {
      await invoke('start_recording');
      console.log('Recording started successfully');
      setIsProcessing(false);
      onRecordingStart();
    } catch (error) {
      console.error('Failed to start recording:', error);
      alert('Failed to start recording. Please check the console for details.');
    } finally {
      setIsStarting(false);
    }
  }, [onRecordingStart, isStarting]);

  const stopRecordingAction = useCallback(async () => {
    console.log('Executing stop recording...');
    try {
      setIsProcessing(true);
      const dataDir = await appDataDir();
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const savePath = `${dataDir}/recording-${timestamp}.wav`;
      
      console.log('Saving recording to:', savePath);
      const result = await invoke('stop_recording', { 
        args: {
          save_path: savePath
        }
      });
      
      setRecordingPath(savePath);
      setIsProcessing(false);
      onRecordingStop();
    } catch (error) {
      console.error('Failed to stop recording:', error);
      if (error instanceof Error) {
        console.error('Error details:', {
          message: error.message,
          name: error.name,
          stack: error.stack,
        });
        if (error.message.includes('No recording in progress')) {
          return;
        }
      } else if (typeof error === 'string' && error.includes('No recording in progress')) {
        return;
      } else if (error && typeof error === 'object' && 'toString' in error) {
        if (error.toString().includes('No recording in progress')) {
          return;
        }
      }
      setIsProcessing(false);
      onRecordingStop();
    } finally {
      setIsStopping(false);
    }
  }, [onRecordingStop]);

  const handleStopRecording = useCallback(async () => {
    if (!isRecording || isStarting || isStopping) return;
    
    console.log('Starting stop countdown...');
    setIsStopping(true);
    setStopCountdown(5);

    // Clear any existing intervals
    if (countdownInterval.current) {
      clearInterval(countdownInterval.current);
      countdownInterval.current = null;
    }

    // Create a controller for the stop action
    const controller = {
      stop: () => {
        if (countdownInterval.current) {
          clearInterval(countdownInterval.current);
          countdownInterval.current = null;
        }
        setIsStopping(false);
        setStopCountdown(5);
      }
    };
    stopTimeoutRef.current = controller;

    // Start countdown
    countdownInterval.current = setInterval(() => {
      setStopCountdown(prev => {
        if (prev <= 1) {
          // Clear interval first
          if (countdownInterval.current) {
            clearInterval(countdownInterval.current);
            countdownInterval.current = null;
          }
          // Schedule stop action
          stopRecordingAction();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
  }, [isRecording, isStarting, isStopping, stopRecordingAction]);

  const cancelStopRecording = useCallback(() => {
    if (stopTimeoutRef.current) {
      stopTimeoutRef.current.stop();
      stopTimeoutRef.current = null;
    }
  }, []);

  useEffect(() => {
    return () => {
      if (countdownInterval.current) clearInterval(countdownInterval.current);
      if (stopTimeoutRef.current) stopTimeoutRef.current.stop();
    };
  }, []);

  const handleGenerateSummary = useCallback(async () => {
    if (!currentTranscript) {
      alert('No transcript available to summarize');
      return;
    }

    try {
      setIsProcessing(true);
      const request: ProcessRequest = {
        transcript: currentTranscript,
        metadata: {
          date: new Date().toISOString(),
          context: {
            meeting_type: 'one_on_one',
            priority: 'medium'
          }
        },
        options: {
          include_sentiment: true,
          include_confidence_score: true,
          format: 'both'
        }
      };

      const response = await fetch('http://localhost:8178/summarize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify(request)
      });

      if (!response.ok) {
        throw new Error(`Summary generation failed: ${response.statusText}`);
      }

      const summaryResponse: SummaryResponse = await response.json();
      if (onSummaryReceived) {
        onSummaryReceived(summaryResponse);
      }
    } catch (error) {
      console.error('Failed to generate summary:', error);
      alert('Failed to generate summary. Check console for details.');
    } finally {
      setIsProcessing(false);
    }
  }, [currentTranscript, onSummaryReceived]);

  return (
    <div className="flex flex-col space-y-2">
      <div className="flex items-center space-x-2 bg-white rounded-full shadow-lg px-4 py-2">
        {isProcessing ? (
          <div className="flex items-center space-x-2">
            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-gray-900"></div>
            <span className="text-sm text-gray-600">Processing...</span>
          </div>
        ) : (
          <>
            {showPlayback ? (
              <>
                <button
                  onClick={handleStartRecording}
                  className="w-10 h-10 flex items-center justify-center bg-red-500 rounded-full text-white hover:bg-red-600 transition-colors"
                >
                  <Mic size={16} />
                </button>

                <div className="w-px h-6 bg-gray-200 mx-1" />

                <div className="flex items-center space-x-1 mx-2">
                  <div className="text-sm text-gray-600 min-w-[40px]">
                    {formatTime(currentTime)}
                  </div>
                  <div 
                    className="relative w-24 h-1 bg-gray-200 rounded-full"
                  >
                    <div 
                      className="absolute h-full bg-blue-500 rounded-full" 
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                  <div className="text-sm text-gray-600 min-w-[40px]">
                    {formatTime(duration)}
                  </div>
                </div>

                <button 
                  className="w-10 h-10 flex items-center justify-center bg-gray-300 rounded-full text-white cursor-not-allowed"
                  disabled
                >
                  <Play size={16} />
                </button>
              </>
            ) : (
              <>
                <button
                  onClick={isRecording ? 
                    (isStopping ? cancelStopRecording : handleStopRecording) : 
                    handleStartRecording}
                  disabled={isStarting || isProcessing || isLoading}
                  className={`w-12 h-12 flex items-center justify-center cursor-pointer ${
                    isStarting || isProcessing || isLoading ? 'bg-gray-400' : 'bg-red-500 hover:bg-red-600'
                  } rounded-full text-white transition-colors relative`}
                >
                  {isLoading ? (
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                  ) : isRecording ? (
                    <>
                      <Square size={20} />
                      {isStopping && (
                        <div className="absolute -top-8 text-red-500 font-medium">
                          {stopCountdown > 0 ? `${stopCountdown}s` : 'Stopping...'}
                        </div>
                      )}
                    </>
                  ) : (
                    <Mic size={20} />
                  )}
                </button>

                <div className="flex items-center space-x-1 mx-4">
                  {barHeights.map((height, index) => (
                    <div
                      key={index}
                      className="w-1 bg-red-500 rounded-full transition-all duration-200"
                      style={{
                        height: isRecording ? height : '4px',
                      }}
                    />
                  ))}
                </div>

                <div className="w-px h-6 bg-gray-200 mx-1" />

                <button
                  onClick={handleGenerateSummary}
                  disabled={!currentTranscript || isProcessing}
                  className={`w-10 h-10 flex items-center justify-center rounded-full transition-colors ${
                    !currentTranscript || isProcessing
                      ? 'bg-gray-300 cursor-not-allowed'
                      : 'bg-blue-500 hover:bg-blue-600 cursor-pointer'
                  } text-white`}
                  title="Generate Summary"
                >
                  <FileText size={16} />
                </button>
              </>
            )}
          </>
        )}
      </div>
      {/* {showPlayback && recordingPath && (
        <div className="text-sm text-gray-600 px-4">
          Recording saved to: {recordingPath}
        </div>
      )} */}
    </div>
  );
};