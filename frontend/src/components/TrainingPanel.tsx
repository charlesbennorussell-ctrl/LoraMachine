import { useState, useCallback, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, Play, Loader2, Trash2, Image as ImageIcon, Settings, CheckCircle2, AlertCircle, AlertTriangle } from 'lucide-react';
import { api } from '../api/client';
import { useStore } from '../store';

export function TrainingPanel() {
  const [config, setConfig] = useState({
    name: '',
    steps: 1000,
    learning_rate: 0.0001,
    batch_size: 1,
    resolution: 1024,
    trigger_word: 'ohwx'
  });
  const [uploadedFiles, setUploadedFiles] = useState<string[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const { status } = useStore();

  // Setup status
  const setupStatus = status.setup;
  const isSetupReady = setupStatus?.ready === true;
  const isSettingUp = setupStatus?.active === true;
  const setupProgress = setupStatus?.progress || 0;
  const setupMessage = setupStatus?.message || 'Not initialized';
  const setupChecks = setupStatus?.checks || {};

  const runSetup = async () => {
    try {
      await api.runSetup();
    } catch (error) {
      console.error('Setup failed to start:', error);
    }
  };

  // Load existing training images on mount
  useEffect(() => {
    const loadImages = async () => {
      try {
        const data = await api.getTrainingImages();
        setUploadedFiles(data.images.map((img: { path: string }) => img.path));
      } catch {
        // Ignore errors
      }
    };
    loadImages();
  }, []);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    setIsUploading(true);
    try {
      const formData = new FormData();
      acceptedFiles.forEach(file => formData.append('files', file));

      const result = await api.uploadTrainingImages(formData);
      setUploadedFiles(prev => [...prev, ...result.uploaded]);
    } catch (error) {
      console.error('Upload failed:', error);
    }
    setIsUploading(false);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.png', '.jpg', '.jpeg', '.webp'] }
  });

  const clearImages = async () => {
    try {
      await api.clearTrainingImages();
      setUploadedFiles([]);
    } catch (error) {
      console.error('Failed to clear images:', error);
    }
  };

  const startTraining = async () => {
    if (!config.name || uploadedFiles.length === 0) return;
    try {
      await api.startTraining(config);
    } catch (error) {
      console.error('Training failed to start:', error);
    }
  };

  const isTraining = status.training?.active;
  const trainingProgress = status.training?.progress || 0;
  const trainingMessage = status.training?.message || 'Idle';

  const getStatusIcon = (checkStatus: string) => {
    switch (checkStatus) {
      case 'ok':
        return <CheckCircle2 className="w-4 h-4 text-emerald-500" />;
      case 'warning':
        return <AlertTriangle className="w-4 h-4 text-amber-500" />;
      case 'error':
        return <AlertCircle className="w-4 h-4 text-red-500" />;
      default:
        return <AlertCircle className="w-4 h-4 text-zinc-500" />;
    }
  };

  return (
    <div className="space-y-8">
      {/* Setup Section - Show if not ready */}
      {!isSetupReady && (
        <section className="p-6 bg-zinc-900/50 border border-zinc-800 rounded-lg">
          <div className="flex items-center gap-3 mb-4">
            <Settings className="w-5 h-5 text-violet-500" />
            <h2 className="text-lg font-medium">System Setup</h2>
          </div>

          {!isSettingUp && setupProgress === 0 && (
            <div className="space-y-4">
              <p className="text-zinc-400 text-sm">
                Before training, we need to download the Flux model (~25GB) and verify your system.
                This only needs to be done once.
              </p>
              <button
                onClick={runSetup}
                className="flex items-center gap-2 px-4 py-2 bg-violet-600 hover:bg-violet-500 rounded-lg font-medium transition-colors"
              >
                <Settings className="w-4 h-4" />
                Run Setup
              </button>
            </div>
          )}

          {(isSettingUp || setupProgress > 0) && (
            <div className="space-y-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-zinc-400">{setupMessage}</span>
                <span className="text-sm font-medium text-zinc-300">{setupProgress}%</span>
              </div>
              <div className="w-full bg-zinc-800 rounded-full h-2 overflow-hidden">
                <div
                  className="bg-gradient-to-r from-violet-500 to-fuchsia-500 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${setupProgress}%` }}
                />
              </div>

              {/* Show check results */}
              {Object.entries(setupChecks).length > 0 && (
                <div className="mt-4 space-y-2">
                  {Object.entries(setupChecks).map(([key, check]) => (
                    <div key={key} className="flex items-center gap-2 text-sm">
                      {getStatusIcon(check.status)}
                      <span className="text-zinc-300 capitalize">{key.replace(/_/g, ' ')}:</span>
                      <span className="text-zinc-500">{check.message}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </section>
      )}

      {/* Setup Complete Banner */}
      {isSetupReady && (
        <section className="p-4 bg-emerald-500/10 border border-emerald-500/30 rounded-lg">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="w-5 h-5 text-emerald-500" />
            <span className="text-emerald-400 font-medium">System ready for training</span>
          </div>
        </section>
      )}

      {/* Upload Section */}
      <section>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-medium">Training Images</h2>
          {uploadedFiles.length > 0 && (
            <button
              onClick={clearImages}
              className="flex items-center gap-1.5 text-sm text-zinc-500 hover:text-red-400 transition-colors"
            >
              <Trash2 className="w-4 h-4" />
              Clear All
            </button>
          )}
        </div>

        <div
          {...getRootProps()}
          className={`border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-all
            ${isDragActive
              ? 'border-violet-500 bg-violet-500/10'
              : 'border-zinc-700 hover:border-zinc-600 hover:bg-zinc-900/50'
            }
            ${isUploading ? 'opacity-50 pointer-events-none' : ''}`}
        >
          <input {...getInputProps()} />
          {isUploading ? (
            <Loader2 className="w-8 h-8 mx-auto mb-3 text-zinc-500 animate-spin" />
          ) : (
            <Upload className="w-8 h-8 mx-auto mb-3 text-zinc-500" />
          )}
          <p className="text-zinc-400">
            {isDragActive
              ? 'Drop images here'
              : isUploading
              ? 'Uploading...'
              : 'Drag & drop training images, or click to select'}
          </p>
          <p className="text-sm text-zinc-600 mt-2">
            Recommended: 10-20 high-quality images of your subject
          </p>
        </div>

        {uploadedFiles.length > 0 && (
          <div className="mt-4">
            <div className="flex items-center gap-2 text-sm text-zinc-400 mb-3">
              <ImageIcon className="w-4 h-4" />
              <span>{uploadedFiles.length} images uploaded</span>
            </div>
            <div className="grid grid-cols-6 gap-2">
              {uploadedFiles.slice(0, 12).map((file, idx) => (
                <div
                  key={idx}
                  className="aspect-square bg-zinc-800 rounded-lg overflow-hidden"
                >
                  <div className="w-full h-full bg-zinc-700 flex items-center justify-center text-xs text-zinc-500">
                    {idx + 1}
                  </div>
                </div>
              ))}
              {uploadedFiles.length > 12 && (
                <div className="aspect-square bg-zinc-800 rounded-lg flex items-center justify-center text-sm text-zinc-500">
                  +{uploadedFiles.length - 12}
                </div>
              )}
            </div>
          </div>
        )}
      </section>

      {/* Config Section */}
      <section>
        <h2 className="text-lg font-medium mb-4">Training Configuration</h2>
        <div className="grid grid-cols-2 gap-4 max-w-2xl">
          <div>
            <label className="block text-sm text-zinc-400 mb-1.5">LoRA Name</label>
            <input
              type="text"
              value={config.name}
              onChange={e => setConfig(c => ({ ...c, name: e.target.value }))}
              placeholder="my-lora"
              className="w-full bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-2 text-sm
                focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-transparent"
            />
          </div>
          <div>
            <label className="block text-sm text-zinc-400 mb-1.5">Trigger Word</label>
            <input
              type="text"
              value={config.trigger_word}
              onChange={e => setConfig(c => ({ ...c, trigger_word: e.target.value }))}
              className="w-full bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-2 text-sm
                focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-transparent"
            />
          </div>
          <div>
            <label className="block text-sm text-zinc-400 mb-1.5">Training Steps</label>
            <input
              type="number"
              value={config.steps}
              onChange={e => setConfig(c => ({ ...c, steps: parseInt(e.target.value) || 1000 }))}
              className="w-full bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-2 text-sm
                focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-transparent"
            />
          </div>
          <div>
            <label className="block text-sm text-zinc-400 mb-1.5">Learning Rate</label>
            <input
              type="number"
              step="0.00001"
              value={config.learning_rate}
              onChange={e => setConfig(c => ({ ...c, learning_rate: parseFloat(e.target.value) || 0.0001 }))}
              className="w-full bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-2 text-sm
                focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-transparent"
            />
          </div>
          <div>
            <label className="block text-sm text-zinc-400 mb-1.5">Resolution</label>
            <select
              value={config.resolution}
              onChange={e => setConfig(c => ({ ...c, resolution: parseInt(e.target.value) }))}
              className="w-full bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-2 text-sm
                focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-transparent"
            >
              <option value={512}>512x512</option>
              <option value={768}>768x768</option>
              <option value={1024}>1024x1024</option>
            </select>
          </div>
          <div>
            <label className="block text-sm text-zinc-400 mb-1.5">Batch Size</label>
            <select
              value={config.batch_size}
              onChange={e => setConfig(c => ({ ...c, batch_size: parseInt(e.target.value) }))}
              className="w-full bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-2 text-sm
                focus:outline-none focus:ring-2 focus:ring-violet-500 focus:border-transparent"
            >
              <option value={1}>1 (16GB VRAM)</option>
              <option value={2}>2 (24GB+ VRAM)</option>
            </select>
          </div>
        </div>
      </section>

      {/* Progress & Start */}
      <section>
        {(isTraining || trainingProgress > 0) && (
          <div className="mb-4 p-4 bg-zinc-900 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm text-zinc-400">{trainingMessage}</span>
              <span className="text-sm font-medium text-zinc-300">{trainingProgress}%</span>
            </div>
            <div className="w-full bg-zinc-800 rounded-full h-2 overflow-hidden">
              <div
                className="bg-gradient-to-r from-violet-500 to-fuchsia-500 h-2 rounded-full transition-all duration-300"
                style={{ width: `${trainingProgress}%` }}
              />
            </div>
          </div>
        )}

        <button
          onClick={startTraining}
          disabled={isTraining || !config.name || uploadedFiles.length === 0}
          className="flex items-center gap-2 px-6 py-2.5 bg-violet-600 hover:bg-violet-500
            disabled:opacity-50 disabled:cursor-not-allowed rounded-lg font-medium transition-colors"
        >
          {isTraining ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Training...
            </>
          ) : (
            <>
              <Play className="w-4 h-4" />
              Start Training
            </>
          )}
        </button>

        {!config.name && uploadedFiles.length > 0 && (
          <p className="mt-2 text-sm text-amber-500">Please enter a name for your LoRA</p>
        )}
        {config.name && uploadedFiles.length === 0 && (
          <p className="mt-2 text-sm text-amber-500">Please upload training images first</p>
        )}
      </section>
    </div>
  );
}
