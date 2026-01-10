import { useState } from 'react';
import { Play, Loader2, Download, Heart, RefreshCw } from 'lucide-react';
import { api } from '../api/client';
import { useStore } from '../store';

export function GenerationGallery() {
  const [config, setConfig] = useState({
    prompt: '',
    negative_prompt: '',
    lora_path: '',
    lora_strength: 0.8,
    steps: 28,
    guidance_scale: 3.5,
    width: 1024,
    height: 1024,
    seed: null as number | null
  });
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedImages, setGeneratedImages] = useState<string[]>([]);
  const { loras } = useStore();

  const generateImage = async () => {
    if (!config.prompt || !config.lora_path) return;

    setIsGenerating(true);
    try {
      const result = await api.generate(config);
      setGeneratedImages(prev => [result.image_path, ...prev]);
    } catch (error) {
      console.error('Generation failed:', error);
    }
    setIsGenerating(false);
  };

  const handleLike = async (imagePath: string) => {
    const imageId = imagePath.split('/').pop() || '';
    try {
      await api.likeImage(imageId);
    } catch (error) {
      console.error('Failed to like image:', error);
    }
  };

  const randomizeSeed = () => {
    setConfig(c => ({ ...c, seed: Math.floor(Math.random() * 2147483647) }));
  };

  return (
    <div className="space-y-8">
      {/* Config */}
      <section className="max-w-3xl">
        <h2 className="text-lg font-medium mb-4">Generation Settings</h2>
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
              placeholder="A photo of ohwx person..."
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
              placeholder="blurry, low quality, distorted..."
              className="w-full bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-2 text-sm
                focus:outline-none focus:ring-2 focus:ring-violet-500"
            />
          </div>

          {/* LoRA Strength Slider */}
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-sm text-zinc-400">LoRA Strength</label>
              <span className="text-sm font-medium text-zinc-300">{config.lora_strength.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.05"
              value={config.lora_strength}
              onChange={e => setConfig(c => ({ ...c, lora_strength: parseFloat(e.target.value) }))}
              className="w-full h-2 bg-zinc-800 rounded-lg appearance-none cursor-pointer
                [&::-webkit-slider-thumb]:appearance-none
                [&::-webkit-slider-thumb]:w-4
                [&::-webkit-slider-thumb]:h-4
                [&::-webkit-slider-thumb]:bg-violet-500
                [&::-webkit-slider-thumb]:rounded-full
                [&::-webkit-slider-thumb]:cursor-pointer"
            />
            <div className="flex justify-between text-xs text-zinc-600 mt-1">
              <span>0.0 (Base Model)</span>
              <span>1.0 (Full LoRA)</span>
            </div>
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
            <label className="block text-sm text-zinc-400 mb-1.5">Seed (optional)</label>
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

        <button
          onClick={generateImage}
          disabled={isGenerating || !config.prompt || !config.lora_path}
          className="mt-6 flex items-center gap-2 px-6 py-2.5 bg-emerald-600 hover:bg-emerald-500
            disabled:opacity-50 disabled:cursor-not-allowed rounded-lg font-medium transition-colors"
        >
          {isGenerating ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Generating...
            </>
          ) : (
            <>
              <Play className="w-4 h-4" />
              Generate
            </>
          )}
        </button>
      </section>

      {/* Generated Images */}
      {generatedImages.length > 0 && (
        <section>
          <h2 className="text-lg font-medium mb-4">Generated Images</h2>
          <div className="grid grid-cols-3 gap-4">
            {generatedImages.map((path, idx) => (
              <div key={idx} className="group relative rounded-lg overflow-hidden bg-zinc-900">
                <img
                  src={api.getImageUrl(path)}
                  alt={`Generated ${idx + 1}`}
                  className="w-full aspect-square object-cover"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent
                  opacity-0 group-hover:opacity-100 transition-opacity">
                  <div className="absolute bottom-0 left-0 right-0 p-3 flex items-center justify-between">
                    <span className="text-xs text-zinc-400">
                      {path.split('/').pop()?.split('_')[0]}
                    </span>
                    <div className="flex gap-2">
                      <button
                        onClick={() => handleLike(path)}
                        className="p-1.5 rounded-full bg-white/20 hover:bg-red-500 transition-colors"
                      >
                        <Heart className="w-4 h-4" />
                      </button>
                      <a
                        href={api.getImageUrl(path)}
                        download
                        className="p-1.5 rounded-full bg-white/20 hover:bg-white/30 transition-colors"
                      >
                        <Download className="w-4 h-4" />
                      </a>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}
