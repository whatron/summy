import { useEffect, useState, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";
import { save } from '@tauri-apps/plugin-dialog';
import { writeTextFile } from '@tauri-apps/plugin-fs';
import { RecordingControls } from "./components/RecordingControls";
import { SummaryResponse } from "./types/summary";
import ReactMarkdown from 'react-markdown';
import { Download, ChevronDown, FileText, FileCode, Play, Edit2, Save } from 'lucide-react';

function App() {
  const [isRecording, setIsRecording] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [barHeights, setBarHeights] = useState(['58%', '76%', '58%']);
  const [error, setError] = useState<string>('');
  const [transcript, setTranscript] = useState<string>('');
  const [isEditing, setIsEditing] = useState(false);
  const [editedTranscript, setEditedTranscript] = useState('');
  const [summary, setSummary] = useState<SummaryResponse | null>(null);
  const transcriptBoxRef = useRef<HTMLDivElement>(null);
  const [showDownloadMenu, setShowDownloadMenu] = useState(false);
  const downloadMenuRef = useRef<HTMLDivElement>(null);
  const [isSendingDummy, setIsSendingDummy] = useState(false);
  const [isSendingBadTranscript, setIsSendingBadTranscript] = useState(false);

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

      // const response = await fetch('http://localhost:8178/process_wav', {
      //   method: 'POST',
      //   body: formData,
      //   headers: {
      //     'Accept': 'application/json'
      //   }
      // });

      // if (!response.ok) {
      //   throw new Error(`Server returned ${response.status}: ${await response.text()}`);
      // }

      // const data = await response.json();
      // if (data.segments && data.segments.length > 0) {
      //   const transcript = data.segments.map((s: any) => s.text).join(' ');
      //   replaceTranscript(transcript);
      // }
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

    // Add the summary text
    sections.push('## Summary\n' + summary.summary);

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
      const dummyTranscript = `Today we'll be exploring the fascinating world of quantum computing, a revolutionary field that's poised to transform how we process information. Let's begin with the fundamental principles that make quantum computing possible. The first key concept we need to understand is quantum superposition. Unlike classical bits that can only be 0 or 1, quantum bits, or qubits, can exist in multiple states simultaneously. This is mathematically described by the wave function ψ(x,t) = Ae^(i(kx-ωt)), which gives us the probability amplitude of finding a particle in a particular state.

Quantum entanglement is another crucial concept. When two or more particles become entangled, their quantum states become correlated, regardless of the distance between them. Einstein famously called this "spooky action at a distance." This phenomenon is essential for quantum teleportation and quantum cryptography. The double-slit experiment demonstrates quantum superposition, where particles create an interference pattern on a screen. Quantum tunneling is another example, where particles can pass through potential barriers that would be impossible to overcome in classical physics.

The quantum circuit model is the most common way to describe quantum computations. A quantum circuit consists of quantum gates that manipulate qubits. The Hadamard gate creates superposition, while the CNOT gate creates entanglement between qubits. Quantum error correction is a critical challenge in quantum computing. Due to decoherence and other quantum effects, qubits are highly susceptible to errors. We use quantum error-correcting codes, such as the surface code, to protect quantum information from these errors.

In terms of real-world applications, quantum computing shows great promise in several areas. In cryptography, it could break current encryption methods and create new quantum-resistant ones. For drug discovery, it enables simulating molecular interactions with unprecedented accuracy. In optimization, it can solve complex logistics and scheduling problems. And in machine learning, it can accelerate certain types of neural network training.

The quantum-classical boundary is an important concept to understand. While quantum computers can solve certain problems exponentially faster than classical computers, they're not universally better. Problems like integer factorization and database search show quantum speedup, while others don't benefit from quantum algorithms. To work in this field, you'll need a basic understanding of linear algebra, familiarity with quantum mechanics principles, knowledge of classical computing fundamentals, and an understanding of probability theory.

The field of quantum computing is rapidly evolving, with major companies like IBM, Google, and Microsoft investing heavily in research and development. IBM's 433-qubit Osprey processor and Google's quantum supremacy experiment are significant milestones in this journey. Looking ahead, we face several key challenges: maintaining quantum coherence, scaling up the number of qubits, reducing error rates, developing new quantum algorithms, and creating a quantum programming ecosystem.

In conclusion, quantum computing represents a paradigm shift in how we process information. While we're still in the early stages of development, the potential applications are vast and transformative. The next decade will likely see significant advances in quantum hardware, algorithms, and applications. Remember that quantum computing is not just about faster computation - it's about solving problems that are currently intractable with classical computers. This includes simulating quantum systems, optimizing complex networks, and breaking current cryptographic systems. As we continue to make progress in this field, we'll need to address these challenges while exploring new applications and use cases. The quantum computing revolution is just beginning, and it's an exciting time to be involved in this field.`;

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

  const handleSendBadTranscript = async () => {
    try {
      setIsSendingBadTranscript(true);
      const badTranscript = `Hay their! Lets tawk about quantim computing and its impakt on the wurld. The furst key consept we knead two understand is quantum supperposition. Unlike clasical bits that can only be zero or won, quantum bits (or q-bits) can exist in multyple states simultainiously. This is mathmaticaly described by the weigh function psi of x and t equals A times e to the eye times k x minus omega t, witch gives us the probibility amplitood of finding a particle in a perticular state.

Quantum entanglemint is another crushal concept. Wen too or more particles become intangled, there quantum states become corelated, regardles of the distance between them. Einstein famusly called this "spookey action at a distance." This phenominon is essential four quantum tellyportation and quantum cryptografy. The quantim circut modle is the most comon way two describe quantum computashuns. A quantum circut consists of quantum gates that manupulate qbits. The Hadamart gate creates supperposition, while the C-NOT gate creates entanglemint between qbits.

In terms of reel-world applicashuns, quantum computing shows promiss in several areas. In cryptografy, it can brake current encrypshun methods. In drug discovry, it helps simulate molekular interacshuns. For optimizashun, it solves complex logistiks problems. And in machine lernin, it helps with neural network traning. The quantim-clasical boundry is an impotent concept two understand. Wile quantum computers can solve certin problems exponenshally faster than clasical computers, their not universaly better.

The feeld of quantum computing is rapidly evolvin, with major companys like IBM, Googel, and Micrsoft investing heavily in reserch and developmint. Looking ahead, we face several key chalenges: maintaining quantum coherense, skaling up the number of qbits, reducing eror rates, developing new algorithims, and creating a quantum programing ekosystem.

In conclushun, quantum computing represents a paradime shift in how we proces informashun. The neckst decade will likely sea significant advances in quantum hardware, algorithims, and applicashuns. Remember that quantum computing is not just about faster computashun - its about solving problems that are currently intractible with clasical computers. As we continue two make progres in this feeld, well need two address these chalenges while exploring new applicashuns and use cases. The quantum computing revolushun is just beginning, and its an exciting time two be involved in this feeld.`;

      // Set the transcript
      replaceTranscript(badTranscript);

      // Send to summarization endpoint
      const response = await fetch('http://localhost:8178/summarize', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({
          transcript: badTranscript,
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
      console.error('Failed to send bad transcript:', error);
      alert('Failed to send bad transcript. Check console for details.');
    } finally {
      setIsSendingBadTranscript(false);
    }
  };

  const handleEditClick = () => {
    setEditedTranscript(transcript);
    setIsEditing(true);
  };

  const handleSaveClick = () => {
    setTranscript(editedTranscript);
    setIsEditing(false);
  };

  const handleCancelClick = () => {
    setIsEditing(false);
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
            <div className="flex items-center space-x-2">
              {isEditing ? (
                <>
                  <button
                    onClick={handleSaveClick}
                    className="flex items-center space-x-1 px-3 py-1 text-sm rounded-md bg-green-500 hover:bg-green-600 text-white transition-colors"
                  >
                    <Save size={16} />
                    <span>Save</span>
                  </button>
                  <button
                    onClick={handleCancelClick}
                    className="flex items-center space-x-1 px-3 py-1 text-sm rounded-md bg-gray-500 hover:bg-gray-600 text-white transition-colors"
                  >
                    <span>Cancel</span>
                  </button>
                </>
              ) : (
                <>
                  <button
                    onClick={handleEditClick}
                    className="flex items-center space-x-1 px-3 py-1 text-sm rounded-md bg-blue-500 hover:bg-blue-600 text-white transition-colors"
                  >
                    <Edit2 size={16} />
                    <span>Edit</span>
                  </button>
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
                  <button
                    onClick={handleSendBadTranscript}
                    disabled={isSendingBadTranscript || !transcript}
                    className={`flex items-center space-x-1 px-3 py-1 text-sm rounded-md transition-colors ${
                      isSendingBadTranscript || !transcript
                        ? 'bg-gray-300 cursor-not-allowed'
                        : 'bg-red-500 hover:bg-red-600 text-white'
                    }`}
                  >
                    <FileText size={16} />
                    <span>{isSendingBadTranscript ? 'Sending...' : 'Send Bad Transcript'}</span>
                  </button>
                </>
              )}
            </div>
          </div>
          {isEditing ? (
            <textarea
              value={editedTranscript}
              onChange={(e) => setEditedTranscript(e.target.value)}
              className="w-full h-[calc(100%-3rem)] p-2 text-gray-600 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              placeholder="Edit your transcript here..."
            />
          ) : (
            <p className="text-gray-600 whitespace-pre-wrap">{transcript || 'No transcript yet...'}</p>
          )}
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
