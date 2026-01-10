import { useState } from 'react';
import { Play, Heart, Loader2, Trash2, RefreshCw } from 'lucide-react';
import { api } from '../api/client';
import { useStore } from '../store';

interface IterationViewerProps {
  onClearImages: () => void;
}

export function IterationViewer({ onClearImages }: IterationViewerProps) {
  const [config, setConfig] = useState({
    prompt: '',
    negative_prompt: '',
    lora_path: '',
    steps: 28,
    guidance_scale: 3.5,
    width: 1024,
    height: 1024,
    seed: null as number | null,
    strength_start: 0.1,
    strength_end: 1.0,
    strength_step: 0.1
  });
  const [isRunning, setIsRunning] = useState(false);
  const { generatedImages, likeImage, loras } = useStore();

  const runIteration = async () => {
    if (!config.prompt || !config.lora_path) return;

    onClearImages(); // Clear previous results
    setIsRunning(true);
    try {
      await api.runIteration(config);
    } catch (error) {
      console.error('Iteration failed:', error);
    }
    setIsRunning(false);
  };

  const handleLike = async (imagePath: string) => {
    const imageId = imagePath.split('/').pop() || '';
    try {
      await api.likeImage(imageId);
      likeImage(imagePath);
    } catch (error) {
      console.error('Failed to like image:', error);
    }
  };

  const randomizeSeed = () => {
    setConfig(c => ({ ...c, seed: Math.floor(Math.random() * 2147483647) }));
  };

  // Filter to only iteration images (those with strength)
  const iterationImages = generatedImages.filter(img => img.strength !== undefined);

  // Calculate how many images are expected
  const expectedCount = Math.round((config.strength_end - config.strength_start) / config.strength_step) + 1;

  return (
    <div className="space-y-8">
      {/* Config */}
      <section className="max-w-3xl">
        <h2 className="text-lg font-medium mb-4">Iteration Settings</h2>
        <div className="space-y-4">
          <div>
            <label className="block text-sm text-zinc-400 mb-1.5">Select LoRA</label>
            <select
              value={config.lora_path}
              onChange={e => setConfig(c => ({ ...c, lora_path: e.target.value }))}
              className="w-full bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-2 text-sm
                focus:outline-none focus:ring-2 focus:ring-violet-500"
            >
              <option value="">Select a trained LoRA...</option>
              {loras.map(lora => (
                <option key={lora.path} value={lora.path}>
                  {lora.name} {!lora.valid && '(invalid)'}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm text-zinc-400 mb-1.5">Prompt</label>
            <textarea
              value={config.prompt}
              onChange={e => setConfig(c => ({ ...c, prompt: e.target.value }))}
              placeholder="A photo of ohwx person in a beautiful garden, golden hour lighting"
              rows={3}
              className="w-full bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-2 text-sm resize-none
                focus:outline-none focus:ring-2 focus:ring-violet-500"
            />
          </div>

          <div>
            <label className="block text-sm text-zinc-400 mb-1.5">Negative Prompt</label>
            <input
              type="text"
              value={config.negative_prompt}
              onChange={e => setConfig(c => ({ ...c, negative_prompt: e.target.value }))}
              placeholder="blurry, low quality, distorted, ugly..."
              className="w-full bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-2 text-sm
                focus:outline-none focus:ring-2 focus:ring-violet-500"
            />
          </div>

          {/* Strength Range */}
          <div className="p-4 bg-zinc-900/50 rounded-lg border border-zinc-800">
            <h3 className="text-sm font-medium text-zinc-300 mb-3">LoRA Strength Range</h3>
            <div className="grid grid-cols-3 gap-4">
              <div>
                <label className="block text-xs text-zinc-500 mb-1">Start</label>
                <input
                  type="number"
                  min="0"
                  max="1"
                  step="0.1"
                  value={config.strength_start}
                  onChange={e => setConfig(c => ({ ...c, strength_start: parseFloat(e.target.value) || 0.1 }))}
                  className="w-full bg-zinc-800 border border-zinc-700 rounded px-2 py-1.5 text-sm
                    focus:outline-none focus:ring-2 focus:ring-violet-500"
                />
              </div>
              <div>
                <label className="block text-xs text-zinc-500 mb-1">End</label>
                <input
                  type="number"
                  min="0"
                  max="1"
                  step="0.1"
                  value={config.strength_end}
                  onChange={e => setConfig(c => ({ ...c, strength_end: parseFloat(e.target.value) || 1.0 }))}
                  className="w-full bg-zinc-800 border border-zinc-700 rounded px-2 py-1.5 text-sm
                    focus:outline-none focus:ring-2 focus:ring-violet-500"
                />
              </div>
              <div>
                <label className="block text-xs text-zinc-500 mb-1">Step</label>
                <input
                  type="number"
                  min="0.05"
                  max="0.5"
                  step="0.05"
                  value={config.strength_step}
                  onChange={e => setConfig(c => ({ ...c, strength_step: parseFloat(e.target.value) || 0.1 }))}
                  className="w-full bg-zinc-800 border border-zinc-700 rounded px-2 py-1.5 text-sm
                    focus:outline-none focus:ring-2 focus:ring-violet-500"
                />
              </div>
            </div>
            <p className="text-xs text-zinc-500 mt-2">
              Will generate {expectedCount} images at strengths: {config.strength_start.toFixed(1)} to {config.strength_end.toFixed(1)}
            </p>
          </div>

          <div className="grid grid-cols-4 gap-4">
            <div>
              <label className="block text-sm text-zinc-400 mb-1.5">Steps</label>
              <input
                type="number"
                value={config.steps}
                onChange={e => setConfig(c => ({ ...c, steps: parseInt(e.target.value) || 28 }))}
                className="w-full bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-2 text-sm
                  focus:outline-none focus:ring-2 focus:ring-violet-500"
              />
            </div>
            <div>
              <label className="block text-sm text-zinc-400 mb-1.5">Guidance</label>
              <input
                type="number"
                step="0.5"
                value={config.guidance_scale}
                onChange={e => setConfig(c => ({ ...c, guidance_scale: parseFloat(e.target.value) || 3.5 }))}
                className="w-full bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-2 text-sm
                  focus:outline-none focus:ring-2 focus:ring-violet-500"
              />
            </div>
            <div>
              <label className="block text-sm text-zinc-400 mb-1.5">Width</label>
              <select
                value={config.width}
                onChange={e => setConfig(c => ({ ...c, width: parseInt(e.target.value) }))}
                className="w-full bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-2 text-sm
                  focus:outline-none focus:ring-2 focus:ring-violet-500"
              >
                <option value={512}>512</option>
                <option value={768}>768</option>
                <option value={1024}>1024</option>
              </select>
            </div>
            <div>
              <label className="block text-sm text-zinc-400 mb-1.5">Height</label>
              <select
                value={config.height}
                onChange={e => setConfig(c => ({ ...c, height: parseInt(e.target.value) }))}
                className="w-full bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-2 text-sm
                  focus:outline-none focus:ring-2 focus:ring-violet-500"
              >
                <option value={512}>512</option>
                <option value={768}>768</option>
                <option value={1024}>1024</option>
              </select>
            </div>
          </div>

          {/* Seed */}
          <div>
            <label className="block text-sm text-zinc-400 mb-1.5">
              Seed <span className="text-zinc-600">(same seed used for all iterations)</span>
            </label>
            <div className="flex gap-2">
              <input
                type="number"
                value={config.seed ?? ''}
                onChange={e => setConfig(c => ({ ...c, seed: e.target.value ? parseInt(e.target.value) : null }))}
                placeholder="Random"
                className="flex-1 bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-2 text-sm
                  focus:outline-none focus:ring-2 focus:ring-violet-500"
              />
              <button
                onClick={randomizeSeed}
                className="px-3 py-2 bg-zinc-800 hover:bg-zinc-700 rounded-lg transition-colors"
              >
                <RefreshCw className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>

        <div className="flex gap-3 mt-6">
          <button
            onClick={runIteration}
            disabled={isRunning || !config.prompt || !config.lora_path}
            className="flex items-center gap-2 px-6 py-2.5 bg-emerald-600 hover:bg-emerald-500
              disabled:opacity-50 disabled:cursor-not-allowed rounded-lg font-medium transition-colors"
          >
            {isRunning ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Generating {iterationImages.length}/{expectedCount}...
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                Run Full Iteration
              </>
            )}
          </button>

          {iterationImages.length > 0 && (
            <button
              onClick={onClearImages}
              className="flex items-center gap-2 px-4 py-2.5 bg-zinc-800 hover:bg-zinc-700
                rounded-lg transition-colors text-zinc-400 hover:text-zinc-200"
            >
              <Trash2 className="w-4 h-4" />
              Clear
            </button>
          )}
        </div>
      </section>

      {/* Results Grid */}
      {iterationImages.length > 0 && (
        <section>
          <h2 className="text-lg font-medium mb-4">
            Iteration Results
            <span className="text-zinc-500 font-normal ml-2">
              ({iterationImages.length} images, LoRA Strength: {config.strength_start.toFixed(1)} to {config.strength_end.toFixed(1)})
            </span>
          </h2>
          <div className="grid grid-cols-5 gap-4">
            {iterationImages
              .sort((a, b) => (a.strength || 0) - (b.strength || 0))
              .map((img, idx) => (
              <div key={idx} className="group relative">
                <img
                  src={api.getImageUrl(img.path)}
                  alt={`Strength ${img.strength}`}
                  className="w-full aspect-square object-cover rounded-lg"
                />
                {/* Strength indicator */}
                <div
                  className="absolute top-2 left-2 px-2 py-0.5 rounded text-xs font-medium"
                  style={{
                    backgroundColor: `hsl(${120 - (img.strength || 0) * 120}, 70%, 40%)`,
                    color: 'white'
                  }}
                >
                  {img.strength?.toFixed(1)}
                </div>

                {/* Hover overlay */}
                <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent
                  opacity-0 group-hover:opacity-100 transition-opacity rounded-lg">
                  <div className="absolute bottom-0 left-0 right-0 p-3 flex items-center justify-between">
                    <span className="text-sm font-medium">
                      Strength: {img.strength?.toFixed(2)}
                    </span>
                    <button
                      onClick={() => handleLike(img.path)}
                      className={`p-1.5 rounded-full transition-colors
                        ${img.liked
                          ? 'bg-red-500 text-white'
                          : 'bg-white/20 hover:bg-red-500 hover:text-white'
                        }`}
                    >
                      <Heart className="w-4 h-4" fill={img.liked ? 'currentColor' : 'none'} />
                    </button>
                  </div>
                </div>

                {/* Liked indicator */}
                {img.liked && (
                  <div className="absolute top-2 right-2">
                    <Heart className="w-5 h-5 text-red-500" fill="currentColor" />
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Strength legend */}
          <div className="mt-6 p-4 bg-zinc-900/50 rounded-lg border border-zinc-800">
            <h3 className="text-sm font-medium text-zinc-300 mb-2">Understanding LoRA Strength</h3>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded" style={{ backgroundColor: 'hsl(120, 70%, 40%)' }} />
                <span className="text-sm text-zinc-400">Low (0.1-0.3): Subtle influence</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded" style={{ backgroundColor: 'hsl(60, 70%, 40%)' }} />
                <span className="text-sm text-zinc-400">Medium (0.4-0.6): Balanced</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded" style={{ backgroundColor: 'hsl(0, 70%, 40%)' }} />
                <span className="text-sm text-zinc-400">High (0.7-1.0): Strong influence</span>
              </div>
            </div>
          </div>
        </section>
      )}
    </div>
  );
}
