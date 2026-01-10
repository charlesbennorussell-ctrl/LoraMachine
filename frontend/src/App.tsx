import { useState, useEffect } from 'react';
import { TrainingPanel } from './components/TrainingPanel';
import { GenerationGallery } from './components/GenerationGallery';
import { IterationViewer } from './components/IterationViewer';
import { RefinementQueue } from './components/RefinementQueue';
import { useWebSocket } from './hooks/useWebSocket';
import { useStore } from './store';
import { api } from './api/client';
import { Cpu, Zap } from 'lucide-react';

type TabType = 'train' | 'generate' | 'iterate' | 'refined';

function App() {
  const [activeTab, setActiveTab] = useState<TabType>('train');
  const {
    setStatus,
    addGeneratedImage,
    addRefinedImage,
    setLoras,
    clearGeneratedImages
  } = useStore();
  const [gpuInfo, setGpuInfo] = useState<{ name: string | null; available: boolean }>({
    name: null,
    available: false
  });

  const { isConnected, lastMessage } = useWebSocket('ws://localhost:8000/ws');

  // Handle WebSocket messages
  useEffect(() => {
    if (lastMessage) {
      switch (lastMessage.type) {
        case 'training_status':
          setStatus('training', lastMessage.data);
          break;
        case 'iteration_progress':
          addGeneratedImage({
            path: lastMessage.data.path,
            strength: lastMessage.data.strength,
            liked: false
          });
          break;
        case 'iteration_complete':
          // Already handled via iteration_progress
          break;
        case 'refinement_complete':
          addRefinedImage(lastMessage.data);
          break;
      }
    }
  }, [lastMessage, setStatus, addGeneratedImage, addRefinedImage]);

  // Fetch initial data
  useEffect(() => {
    const fetchData = async () => {
      try {
        const status = await api.getStatus();
        setGpuInfo({
          name: status.gpu_name,
          available: status.gpu_available
        });

        const lorasData = await api.getLoras();
        setLoras(lorasData.loras);
      } catch (error) {
        console.error('Failed to fetch initial data:', error);
      }
    };

    fetchData();

    // Refresh LoRAs periodically
    const interval = setInterval(async () => {
      try {
        const lorasData = await api.getLoras();
        setLoras(lorasData.loras);
      } catch {
        // Ignore errors
      }
    }, 10000);

    return () => clearInterval(interval);
  }, [setLoras]);

  const tabs: { id: TabType; label: string }[] = [
    { id: 'train', label: 'Train' },
    { id: 'generate', label: 'Generate' },
    { id: 'iterate', label: 'Iterate' },
    { id: 'refined', label: 'Refined' }
  ];

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100">
      {/* Header */}
      <header className="border-b border-zinc-800 px-6 py-4">
        <div className="flex items-center justify-between max-w-7xl mx-auto">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-gradient-to-br from-violet-500 to-fuchsia-500 rounded-lg flex items-center justify-center">
              <Zap className="w-5 h-5 text-white" />
            </div>
            <h1 className="text-xl font-semibold tracking-tight">
              Flux LoRA Pipeline
            </h1>
          </div>
          <div className="flex items-center gap-4">
            {/* GPU Status */}
            <div className="flex items-center gap-2 px-3 py-1.5 bg-zinc-900 rounded-lg">
              <Cpu className="w-4 h-4 text-zinc-500" />
              <span className="text-sm text-zinc-400">
                {gpuInfo.available ? gpuInfo.name : 'No GPU'}
              </span>
            </div>
            {/* WebSocket Status */}
            <div className="flex items-center gap-2">
              <span className={`w-2 h-2 rounded-full ${isConnected ? 'bg-emerald-500' : 'bg-red-500'}`} />
              <span className="text-sm text-zinc-500">
                {isConnected ? 'Connected' : 'Disconnected'}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="border-b border-zinc-800">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex gap-1">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-4 py-3 text-sm font-medium transition-colors relative
                  ${activeTab === tab.id
                    ? 'text-white'
                    : 'text-zinc-500 hover:text-zinc-300'
                  }`}
              >
                {tab.label}
                {activeTab === tab.id && (
                  <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-violet-500" />
                )}
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        {activeTab === 'train' && <TrainingPanel />}
        {activeTab === 'generate' && <GenerationGallery />}
        {activeTab === 'iterate' && <IterationViewer onClearImages={clearGeneratedImages} />}
        {activeTab === 'refined' && <RefinementQueue />}
      </main>
    </div>
  );
}

export default App;
