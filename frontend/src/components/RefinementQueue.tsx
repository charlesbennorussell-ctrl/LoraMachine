import { useEffect, useState } from 'react';
import { Sparkles, Download, ExternalLink, RefreshCw } from 'lucide-react';
import { useStore } from '../store';
import { api } from '../api/client';

export function RefinementQueue() {
  const { refinedImages } = useStore();
  const [loadedImages, setLoadedImages] = useState<Array<{ original: string; refined: string }>>([]);
  const [isLoading, setIsLoading] = useState(false);

  // Combine store refined images with loaded ones
  const allRefinedImages = [...refinedImages, ...loadedImages];

  const loadExistingImages = async () => {
    setIsLoading(true);
    try {
      const data = await api.getRefinedImages();
      setLoadedImages(data.images.filter((img: { refined: string; original: string | null }) =>
        img.refined && img.original
      ).map((img: { refined: string; original: string }) => ({
        original: img.original,
        refined: img.refined
      })));
    } catch (error) {
      console.error('Failed to load refined images:', error);
    }
    setIsLoading(false);
  };

  useEffect(() => {
    loadExistingImages();
  }, []);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-amber-500" />
          <h2 className="text-lg font-medium">Auto-Refined Images</h2>
          <span className="text-sm text-zinc-500">
            ({allRefinedImages.length} images)
          </span>
        </div>
        <button
          onClick={loadExistingImages}
          disabled={isLoading}
          className="flex items-center gap-2 px-3 py-1.5 text-sm text-zinc-400 hover:text-zinc-200
            bg-zinc-800 hover:bg-zinc-700 rounded-lg transition-colors"
        >
          <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {allRefinedImages.length === 0 ? (
        <div className="text-center py-16 border-2 border-dashed border-zinc-800 rounded-xl">
          <Sparkles className="w-12 h-12 mx-auto mb-4 text-zinc-700" />
          <p className="text-zinc-500 mb-2">No refined images yet</p>
          <p className="text-sm text-zinc-600">
            Like images in the Iterate view to auto-refine them
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-2 gap-6">
          {allRefinedImages.map((item, idx) => (
            <div key={idx} className="bg-zinc-900 rounded-xl overflow-hidden border border-zinc-800">
              <div className="grid grid-cols-2">
                {/* Original */}
                <div className="relative">
                  {item.original ? (
                    <img
                      src={api.getImageUrl(item.original)}
                      alt="Original"
                      className="w-full aspect-square object-cover"
                    />
                  ) : (
                    <div className="w-full aspect-square bg-zinc-800 flex items-center justify-center">
                      <span className="text-zinc-600 text-sm">Original not found</span>
                    </div>
                  )}
                  <span className="absolute bottom-2 left-2 px-2 py-0.5 bg-black/60 rounded text-xs">
                    Original
                  </span>
                </div>

                {/* Refined */}
                <div className="relative">
                  <img
                    src={api.getImageUrl(item.refined)}
                    alt="Refined"
                    className="w-full aspect-square object-cover"
                  />
                  <span className="absolute bottom-2 left-2 px-2 py-0.5 bg-amber-500/80 rounded text-xs font-medium">
                    Refined
                  </span>
                </div>
              </div>

              {/* Actions */}
              <div className="p-3 border-t border-zinc-800 flex justify-between items-center">
                <span className="text-xs text-zinc-500">
                  {item.refined.split('/').pop()?.slice(0, 20)}...
                </span>
                <div className="flex gap-2">
                  <a
                    href={api.getImageUrl(item.refined)}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center gap-1.5 text-sm text-zinc-400 hover:text-white transition-colors"
                  >
                    <ExternalLink className="w-4 h-4" />
                    View
                  </a>
                  <a
                    href={api.getImageUrl(item.refined)}
                    download
                    className="flex items-center gap-1.5 text-sm text-zinc-400 hover:text-white transition-colors"
                  >
                    <Download className="w-4 h-4" />
                    Download
                  </a>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Info Panel */}
      <div className="p-4 bg-zinc-900/50 rounded-lg border border-zinc-800">
        <h3 className="text-sm font-medium text-zinc-300 mb-2">About Refinement</h3>
        <p className="text-sm text-zinc-500">
          When you like an image in the Iterate view, it's automatically queued for refinement.
          The refinement process uses AI upscaling to enhance image quality and detail.
          Refined images are typically 2-4x the original resolution.
        </p>
      </div>
    </div>
  );
}
