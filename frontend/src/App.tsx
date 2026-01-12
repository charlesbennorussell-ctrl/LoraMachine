import { useState, useEffect } from 'react';
import { TrainingPanel } from './components/TrainingPanel';
import { GenerationGallery } from './components/GenerationGallery';
import { IterationViewer } from './components/IterationViewer';
import { RefinementQueue } from './components/RefinementQueue';
import { LoRAManager } from './components/LoRAManager';
import { ImageGrid } from './components/ImageGrid';
import { useWebSocket } from './hooks/useWebSocket';
import { useStore } from './store';
import { api } from './api/client';
import { Cpu, Zap, Grid3X3 } from 'lucide-react';

type TabType = 'train' | 'generate' | 'iterate' | 'refined' | 'loras';

function App() {
  const [activeTab, setActiveTab] = useState<TabType>('iterate');
  const [selectedGridImage, setSelectedGridImage] = useState<string | null>(null);
  const {
    setStatus,
    setSetupStatus,
    addGeneratedImage,
    addImg2ImgImage,
    addRefinedImage,
    setLoras,
    clearImg2ImgImages,
    setInputImage,
    img2imgImages,
    generatedImages
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
        case 'setup_status':
          setSetupStatus(lastMessage.data);
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
        case 'img2img_iteration_progress':
          addImg2ImgImage({
            path: lastMessage.data.path,
            creativity: lastMessage.data.creativity,
            liked: false
          });
          break;
        case 'img2img_iteration_complete':
          // Already handled via img2img_iteration_progress
          break;
        case 'img2img_iteration_error':
          console.error('Img2img iteration error:', lastMessage.data.error);
          break;
        case 'refinement_complete':
          addRefinedImage(lastMessage.data);
          break;
      }
    }
  }, [lastMessage, setStatus, setSetupStatus, addGeneratedImage, addImg2ImgImage, addRefinedImage]);

  // Fetch initial data
  useEffect(() => {
    const fetchData = async () => {
      try {
        const status = await api.getStatus();
        setGpuInfo({
          name: status.gpu_name,
          available: status.gpu_available
        });

        // Load setup status if available
        if (status.setup) {
          setSetupStatus(status.setup);
        }

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
    { id: 'refined', label: 'Refined' },
    { id: 'loras', label: 'LoRAs' }
  ];

  return (
    <div className="h-screen flex flex-col bg-zinc-950 text-zinc-100 overflow-hidden">
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

      {/* Main Content - 2 Column Layout (30% controls, 70% grid) */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Column - Controls (30%) */}
        <main className="w-[30%] min-w-[320px] overflow-y-auto px-4 py-6 border-r border-zinc-800">
          <div>
            {activeTab === 'train' && <TrainingPanel />}
            {activeTab === 'generate' && <GenerationGallery />}
            {activeTab === 'iterate' && (
              <IterationViewer
                onClearImages={clearImg2ImgImages}
                selectedImage={selectedGridImage}
                onUseSelectedImage={(path) => {
                  // Convert grid image to input image for iteration
                  setInputImage({
                    path: path,
                    url: `http://localhost:8000${path}`
                  });
                  setSelectedGridImage(null);
                }}
              />
            )}
            {activeTab === 'refined' && <RefinementQueue />}
            {activeTab === 'loras' && <LoRAManager />}
          </div>
        </main>

        {/* Right Column - Image Grid (Always visible) */}
        <aside className="w-80 border-l border-zinc-800 bg-zinc-900/50 flex flex-col">
          <div className="flex items-center gap-2 px-4 py-3 border-b border-zinc-800">
            <Grid3X3 className="w-4 h-4 text-zinc-500" />
            <span className="text-sm font-medium text-zinc-300">
              Results ({img2imgImages.length + generatedImages.length})
            </span>
          </div>
          <div className="flex-1 overflow-hidden">
            <ImageGrid
              onSelectImage={(path) => {
                setSelectedGridImage(path);
                // If on iterate tab, also set as input
                if (activeTab === 'iterate') {
                  setInputImage({
                    path: path,
                    url: `http://localhost:8000${path}`
                  });
                }
              }}
              selectedImage={selectedGridImage}
            />
          </div>
        </aside>
      </div>
    </div>
  );
}

export default App;
