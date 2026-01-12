import { useState, useCallback, useEffect } from 'react';
import { Play, Heart, Loader2, Trash2, RefreshCw, Upload, X, ImageIcon } from 'lucide-react';
import { useDropzone } from 'react-dropzone';
import { api } from '../api/client';
import { useStore } from '../store';

interface IterationViewerProps {
  onClearImages: () => void;
  selectedImage?: string | null;
  onUseSelectedImage?: (path: string) => void;
}

export function IterationViewer({ onClearImages, selectedImage, onUseSelectedImage }: IterationViewerProps) {
  const [config, setConfig] = useState({
    prompt: '',
    negative_prompt: '',
    lora_path: '',
    lora_strength: 1.0,  // Full LoRA influence
    steps: 28,
    guidance_scale: 3.5,
    width: 1024,
    height: 1024,
    seed: null as number | null,
    // 5 versions at different creativity levels (0.0 doesn't work - results in 0 steps)
    creativity_values: [0.1, 0.3, 0.5, 0.7, 0.9]
  });
  const [isRunning, setIsRunning] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [iterationImages, setIterationImages] = useState<Array<{ path: string; creativity: number; liked?: boolean }>>([]);
  const { likeImage, loras, inputImage, setInputImage, clearImg2ImgImages, img2imgImages } = useStore();

  // Sync local state with store (WebSocket updates go to store via App.tsx)
  useEffect(() => {
    if (img2imgImages.length > 0 && isRunning) {
      setIterationImages(img2imgImages.map(img => ({
        path: img.path,
        creativity: img.creativity || 0,
        liked: img.liked
      })));
      // Check if iteration is complete
      if (img2imgImages.length >= config.creativity_values.length) {
        setIsRunning(false);
      }
    }
  }, [img2imgImages, isRunning, config.creativity_values.length]);

  // Handle input image upload
  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;

    setIsUploading(true);
    try {
      const result = await api.uploadInputImage(acceptedFiles[0]);
      setInputImage({ path: result.path, url: result.url });
    } catch (error) {
      console.error('Failed to upload input image:', error);
    }
    setIsUploading(false);
  }, [setInputImage]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.webp']
    },
    maxFiles: 1
  });

  const runIteration = async () => {
    if (!config.prompt || !config.lora_path || !inputImage) return;

    setIterationImages([]); // Clear previous results
    clearImg2ImgImages();
    setIsRunning(true);

    try {
      await api.runImg2ImgIteration({
        prompt: config.prompt,
        negative_prompt: config.negative_prompt,
        lora_path: config.lora_path,
        lora_strength: config.lora_strength,
        steps: config.steps,
        guidance_scale: config.guidance_scale,
        width: config.width,
        height: config.height,
        seed: config.seed,
        creativity_values: config.creativity_values
      }, inputImage.path);
    } catch (error) {
      console.error('Iteration failed:', error);
      setIsRunning(false);
    }
  };

  // Note: WebSocket updates are handled in App.tsx via useWebSocket hook
  // This component uses the store to receive updates

  const handleLike = async (imagePath: string) => {
    const imageId = imagePath.split('/').pop() || '';
    try {
      await api.likeImage(imageId);
      likeImage(imagePath);
      setIterationImages(prev =>
        prev.map(img => img.path === imagePath ? { ...img, liked: true } : img)
      );
    } catch (error) {
      console.error('Failed to like image:', error);
    }
  };

  const randomizeSeed = () => {
    setConfig(c => ({ ...c, seed: Math.floor(Math.random() * 2147483647) }));
  };

  const clearInputImage = () => {
    setInputImage(null);
  };

  const clearResults = () => {
    setIterationImages([]);
    clearImg2ImgImages();
    onClearImages();
  };

  // Sort by creativity
  const sortedImages = [...iterationImages].sort((a, b) => a.creativity - b.creativity);

  return (
    <div className="space-y-8">
      {/* Input Image Upload */}
      <section className="max-w-3xl">
        <h2 className="text-lg font-medium mb-4">Iterate Creativity Levels (Img2Img)</h2>
        <p className="text-sm text-zinc-400 mb-4">
          Generate 5 versions of your input image at different creativity levels (0.0, 0.2, 0.4, 0.6, 0.8).
          Lower creativity stays closer to your input, higher allows more transformation.
        </p>

        {!inputImage ? (
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-xl p-8 text-center cursor-pointer transition-colors
              ${isDragActive
                ? 'border-violet-500 bg-violet-500/10'
                : 'border-zinc-700 hover:border-zinc-500 bg-zinc-900/50'
              }`}
          >
            <input {...getInputProps()} />
            {isUploading ? (
              <div className="flex flex-col items-center gap-3">
                <Loader2 className="w-8 h-8 animate-spin text-violet-500" />
                <p className="text-zinc-400">Uploading...</p>
              </div>
            ) : (
              <div className="flex flex-col items-center gap-3">
                <Upload className="w-8 h-8 text-zinc-500" />
                <p className="text-zinc-300">
                  {isDragActive ? 'Drop image here' : 'Drag & drop an input image, or click to browse'}
                </p>
                <p className="text-xs text-zinc-500">PNG, JPG, WEBP up to 10MB</p>
              </div>
            )}
          </div>
        ) : (
          <div className="relative inline-block">
            <img
              src={api.getImageUrl(inputImage.url)}
              alt="Input image"
              className="max-w-sm rounded-lg border border-zinc-700"
            />
            <button
              onClick={clearInputImage}
              className="absolute -top-2 -right-2 p-1 bg-red-500 hover:bg-red-400 rounded-full transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
            <div className="absolute bottom-2 left-2 px-2 py-1 bg-black/70 rounded text-xs text-zinc-300">
              Input Image
            </div>
          </div>
        )}
      </section>

      {/* Config */}
      <section className="max-w-3xl">
        <h3 className="text-md font-medium mb-4">Iteration Settings</h3>
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

          {/* Creativity Levels Preview */}
          <div className="p-4 bg-violet-900/20 border border-violet-700/30 rounded-lg">
            <h3 className="text-sm font-medium text-violet-300 mb-3">Creativity Sweep</h3>
            <div className="flex gap-2 mb-2">
              {config.creativity_values.map((val, idx) => (
                <div
                  key={idx}
                  className="flex-1 text-center py-2 rounded bg-zinc-800 text-sm font-medium"
                  style={{
                    backgroundColor: `hsla(${260 - val * 100}, 70%, 30%, 0.5)`
                  }}
                >
                  {val.toFixed(1)}
                </div>
              ))}
            </div>
            <p className="text-xs text-zinc-400">
              Will generate 5 images at creativity levels: 0.1 (subtle) to 0.9 (creative)
            </p>
          </div>

          {/* LoRA Strength Slider */}
          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-sm text-zinc-400">LoRA Strength (fixed at full for all)</label>
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
            disabled={isRunning || !config.prompt || !config.lora_path || !inputImage}
            className="flex items-center gap-2 px-6 py-2.5 bg-emerald-600 hover:bg-emerald-500
              disabled:opacity-50 disabled:cursor-not-allowed rounded-lg font-medium transition-colors"
          >
            {isRunning ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Generating {iterationImages.length}/{config.creativity_values.length}...
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                Run 5 Iterations
              </>
            )}
          </button>

          {iterationImages.length > 0 && (
            <button
              onClick={clearResults}
              className="flex items-center gap-2 px-4 py-2.5 bg-zinc-800 hover:bg-zinc-700
                rounded-lg transition-colors text-zinc-400 hover:text-zinc-200"
            >
              <Trash2 className="w-4 h-4" />
              Clear
            </button>
          )}
        </div>

        {!inputImage && (
          <p className="text-sm text-amber-400 mt-2">
            <ImageIcon className="w-4 h-4 inline mr-1" />
            Upload an input image above to enable iteration
          </p>
        )}
      </section>

      {/* Results Grid */}
      {sortedImages.length > 0 && (
        <section>
          <h2 className="text-lg font-medium mb-4">
            Creativity Sweep Results
            <span className="text-zinc-500 font-normal ml-2">
              ({sortedImages.length} images)
            </span>
          </h2>
          <div className="grid grid-cols-5 gap-4">
            {sortedImages.map((img, idx) => (
              <div key={idx} className="group relative">
                <img
                  src={api.getImageUrl(img.path)}
                  alt={`Creativity ${img.creativity}`}
                  className="w-full aspect-square object-cover rounded-lg"
                />
                {/* Creativity indicator */}
                <div
                  className="absolute top-2 left-2 px-2 py-0.5 rounded text-xs font-medium"
                  style={{
                    backgroundColor: `hsla(${260 - img.creativity * 200}, 70%, 40%, 0.9)`,
                    color: 'white'
                  }}
                >
                  {img.creativity.toFixed(1)}
                </div>

                {/* Hover overlay */}
                <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent
                  opacity-0 group-hover:opacity-100 transition-opacity rounded-lg">
                  <div className="absolute bottom-0 left-0 right-0 p-3 flex items-center justify-between">
                    <span className="text-sm font-medium">
                      Creativity: {img.creativity.toFixed(2)}
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

          {/* Creativity legend */}
          <div className="mt-6 p-4 bg-zinc-900/50 rounded-lg border border-zinc-800">
            <h3 className="text-sm font-medium text-zinc-300 mb-2">Understanding Creativity Levels</h3>
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded" style={{ backgroundColor: 'hsla(260, 70%, 40%, 0.9)' }} />
                <span className="text-sm text-zinc-400">0.1: Very close to input</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded" style={{ backgroundColor: 'hsla(200, 70%, 40%, 0.9)' }} />
                <span className="text-sm text-zinc-400">0.5: Balanced transformation</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 rounded" style={{ backgroundColor: 'hsla(100, 70%, 40%, 0.9)' }} />
                <span className="text-sm text-zinc-400">0.9: More creative freedom</span>
              </div>
            </div>
          </div>
        </section>
      )}
    </div>
  );
}
