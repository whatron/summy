import { useEffect, useState, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { save } from '@tauri-apps/plugin-dialog';
import { writeTextFile } from '@tauri-apps/plugin-fs';
import { RecordingControls } from "./components/RecordingControls";
import { SummaryResponse } from "./types/summary";
import ReactMarkdown from 'react-markdown';
import { Download, ChevronDown, FileText, FileCode, Play } from 'lucide-react';

function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [barHeights, setBarHeights] = useState(['58%', '76%', '58%']);
  const [error, setError] = useState<string>('');
  const [transcript, setTranscript] = useState<string>('');
  const [summary, setSummary] = useState<SummaryResponse | null>(null);
  const transcriptBoxRef = useRef<HTMLDivElement>(null);
  const [showDownloadMenu, setShowDownloadMenu] = useState(false);
  const downloadMenuRef = useRef<HTMLDivElement>(null);
  const [isSendingDummy, setIsSendingDummy] = useState(false);

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

  // Close download menu when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (downloadMenuRef.current && !downloadMenuRef.current.contains(event.target as Node)) {
        setShowDownloadMenu(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

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
        body: formData,
        headers: {
          'Accept': 'application/json'
        }
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

  const handleSummaryReceived = (newSummary: SummaryResponse) => {
    setSummary(newSummary);
  };

  const formatSummaryAsMarkdown = (summary: SummaryResponse) => {
    const sections = [];

    // Add lecture metadata
    if (summary.lecture_metadata) {
      sections.push('## Lecture Information\n' +
        `- Subject: ${summary.lecture_metadata.subject}\n` +
        `- Topic: ${summary.lecture_metadata.topic}\n` +
        `- Level: ${summary.lecture_metadata.level}\n` +
        `- Duration: ${summary.lecture_metadata.estimated_duration}`
      );
    }

    if (summary.summary.main_concepts?.length > 0) {
      sections.push('## Main Concepts\n' + summary.summary.main_concepts.map(concept => `- ${concept}`).join('\n'));
    }

    if (summary.summary.key_definitions?.length > 0) {
      sections.push('## Key Definitions\n' + summary.summary.key_definitions.map(def => `- ${def}`).join('\n'));
    }

    if (summary.summary.important_formulas?.length > 0) {
      sections.push('## Important Formulas\n' + summary.summary.important_formulas.map(formula => `- ${formula}`).join('\n'));
    }

    if (summary.summary.examples?.length > 0) {
      sections.push('## Examples\n' + summary.summary.examples.map(example => `- ${example}`).join('\n'));
    }

    if (summary.summary.learning_objectives?.length > 0) {
      sections.push('## Learning Objectives\n' + summary.summary.learning_objectives.map(objective => `- ${objective}`).join('\n'));
    }

    if (summary.summary.prerequisites?.length > 0) {
      sections.push('## Prerequisites\n' + summary.summary.prerequisites.map(prereq => `- ${prereq}`).join('\n'));
    }

    // Add difficulty and confidence information
    sections.push('## Additional Information\n' +
      `- Difficulty Level: ${summary.summary.difficulty_level}\n` +
      `- Confidence Score: ${(summary.summary.confidence_score * 100).toFixed(1)}%\n` +
      `- Sentiment: ${summary.summary.sentiment}`
    );

    return sections.join('\n\n');
  };

  const downloadText = async (content: string, suggestedName: string) => {
    try {
      // Open save dialog
      const filePath = await save({
        filters: [{
          name: 'Text Files',
          extensions: ['txt', 'md']
        }],
        defaultPath: suggestedName
      });

      if (filePath) {
        // Write the file
        await writeTextFile(filePath, content);
      }
    } catch (error) {
      console.error('Failed to save file:', error);
      alert('Failed to save file. Check console for details.');
    }
  };

  const handleDownload = async (type: 'transcript' | 'summary-md' | 'summary-txt') => {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    
    try {
      switch (type) {
        case 'transcript':
          await downloadText(transcript, `transcript-${timestamp}.txt`);
          break;
        case 'summary-md':
          if (summary) {
            await downloadText(formatSummaryAsMarkdown(summary), `summary-${timestamp}.md`);
          }
          break;
        case 'summary-txt':
          if (summary) {
            // Convert markdown to plain text by removing markdown syntax
            const plainText = formatSummaryAsMarkdown(summary)
              .replace(/^##\s+/gm, '') // Remove ## headers
              .replace(/^- /gm, '• ') // Convert - to bullet points
              .replace(/\n\n/g, '\n'); // Remove extra newlines
            await downloadText(plainText, `summary-${timestamp}.txt`);
          }
          break;
      }
    } catch (error) {
      console.error('Failed to download:', error);
      alert('Failed to download file. Check console for details.');
    } finally {
      setShowDownloadMenu(false);
    }
  };

  const handleSendDummyTranscript = async () => {
    try {
      setIsSendingDummy(true);
      const dummyTranscript = `Today we'll be discussing the fascinating world of quantum mechanics. Let's start with the fundamental principles that govern the behavior of particles at the quantum level.

The first key concept is wave-particle duality, which states that particles can exhibit both wave-like and particle-like properties. This was first demonstrated in the famous double-slit experiment, where electrons showed interference patterns similar to waves.

The Heisenberg Uncertainty Principle is another crucial concept. It tells us that we cannot simultaneously know both the position and momentum of a particle with perfect precision. This is mathematically expressed as ΔxΔp ≥ ħ/2, where ħ is the reduced Planck constant.

Quantum superposition is the ability of quantum systems to exist in multiple states simultaneously. This is described by the wave function ψ(x,t) = Ae^(i(kx-ωt)), which gives us the probability amplitude of finding a particle at a particular position and time.

Let's look at some practical examples. The double-slit experiment demonstrates wave-particle duality, where electrons create an interference pattern on a screen. Quantum tunneling is another example, where particles can pass through potential barriers that would be impossible to overcome in classical physics.

To understand these concepts, you'll need a basic understanding of classical mechanics, familiarity with wave phenomena, and knowledge of differential equations. The key learning objectives for this lecture are to understand the fundamental principles of quantum mechanics, apply quantum concepts to solve basic problems, and recognize quantum effects in real-world applications.`;

      // Set the transcript
      replaceTranscript(dummyTranscript);

      // Send to summarization endpoint
      const response = await fetch('http://localhost:8178/summarize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({
          transcript: dummyTranscript,
          metadata: {
            date: new Date().toISOString(),
            context: {
              meeting_type: 'lecture',
              priority: 'high'
            }
          },
          options: {
            include_sentiment: true,
            include_confidence_score: true,
            format: 'both'
          }
        })
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}: ${await response.text()}`);
      }

      const summaryResponse: SummaryResponse = await response.json();
      handleSummaryReceived(summaryResponse);
    } catch (error) {
      console.error('Failed to send dummy transcript:', error);
      alert('Failed to send dummy transcript. Check console for details.');
    } finally {
      setIsSendingDummy(false);
    }
  };

  return (
    <main className="container">
      {/* Fixed transcript box */}
      <div className="absolute bottom-32 left-1/2 transform -translate-x-1/2 w-4/5 max-w-4xl">
        <div 
          className="bg-white rounded-lg p-4 h-64 overflow-y-auto border-2 border-gray-300 shadow-[0_4px_12px_rgba(0,0,0,0.15)]" 
          ref={transcriptBoxRef}
        >
          <div className="flex justify-between items-center mb-2 sticky top-0 bg-white pb-2 border-b border-gray-200">
            <h2 className="text-lg font-semibold text-gray-800">Transcript</h2>
            <button
              onClick={handleSendDummyTranscript}
              disabled={isSendingDummy}
              className={`flex items-center space-x-1 px-3 py-1 text-sm rounded-md transition-colors ${
                isSendingDummy
                  ? 'bg-gray-300 cursor-not-allowed'
                  : 'bg-blue-500 hover:bg-blue-600 text-white'
              }`}
            >
              <Play size={16} />
              <span>{isSendingDummy ? 'Sending...' : 'Send Dummy Lecture'}</span>
            </button>
          </div>
          <p className="text-gray-600 whitespace-pre-wrap">{transcript || 'No transcript yet...'}</p>
        </div>
      </div>

      {/* Summary box */}
      {summary && (
        <div className="absolute bottom-96 left-1/2 transform -translate-x-1/2 w-4/5 max-w-4xl">
          <div className="bg-white rounded-lg p-4 border-2 border-gray-300 shadow-[0_4px_12px_rgba(0,0,0,0.15)]">
            <div className="flex justify-between items-center mb-2 sticky top-0 bg-white pb-2 border-b border-gray-200 z-10">
              <h2 className="text-lg font-semibold text-gray-800">Summary</h2>
              <div className="relative" ref={downloadMenuRef}>
                <button
                  onClick={() => setShowDownloadMenu(!showDownloadMenu)}
                  className="flex items-center space-x-1 px-3 py-1 text-sm text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-md transition-colors"
                >
                  <Download size={16} />
                  <span>Download</span>
                  <ChevronDown size={14} className={`transition-transform ${showDownloadMenu ? 'rotate-180' : ''}`} />
                </button>
                
                {showDownloadMenu && (
                  <div className="absolute right-0 mt-1 w-48 bg-white rounded-md shadow-lg border border-gray-200 py-1 z-20">
                    <button
                      onClick={() => handleDownload('transcript')}
                      className="w-full px-4 py-2 text-left text-sm text-gray-700 hover:bg-gray-100 flex items-center space-x-2"
                    >
                      <FileText size={14} />
                      <span>Download Transcript</span>
                    </button>
                    <button
                      onClick={() => handleDownload('summary-md')}
                      className="w-full px-4 py-2 text-left text-sm text-gray-700 hover:bg-gray-100 flex items-center space-x-2"
                    >
                      <FileCode size={14} />
                      <span>Download Summary (Markdown)</span>
                    </button>
                    <button
                      onClick={() => handleDownload('summary-txt')}
                      className="w-full px-4 py-2 text-left text-sm text-gray-700 hover:bg-gray-100 flex items-center space-x-2"
                    >
                      <FileText size={14} />
                      <span>Download Summary (Text)</span>
                    </button>
                  </div>
                )}
              </div>
            </div>
            <div className="prose prose-sm max-w-none h-96 overflow-y-auto pr-2">
              <ReactMarkdown>
                {formatSummaryAsMarkdown(summary)}
              </ReactMarkdown>
            </div>
          </div>
        </div>
      )}

      {/* Recording controls */}
      <div className="absolute bottom-16 left-1/2 transform -translate-x-1/2 z-10">
        <div className="bg-white rounded-full shadow-lg flex items-center">
          <RecordingControls
            isRecording={isRecording}
            isLoading={isLoading}
            onRecordingStop={handleRecordingStop}
            onRecordingStart={handleRecordingStart}
            onTranscriptReceived={handleTranscriptReceived}
            onSummaryReceived={handleSummaryReceived}
            currentTranscript={transcript}
            barHeights={barHeights}
          />
        </div>
      </div>
    </main>
  );
}

export default App;
