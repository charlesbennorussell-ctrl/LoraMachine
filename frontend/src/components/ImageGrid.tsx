import { useState } from 'react';
import { Heart, Trash2, Download, X, ZoomIn } from 'lucide-react';
import { useStore, GeneratedImage } from '../store';
import { api } from '../api/client';

interface ImageGridProps {
  onSelectImage?: (imagePath: string) => void;
  selectedImage?: string | null;
}

export function ImageGrid({ onSelectImage, selectedImage }: ImageGridProps) {
  const { img2imgImages, generatedImages, likeImage, removeImage } = useStore();
  const [lightboxImage, setLightboxImage] = useState<string | null>(null);
  const [hoveredImage, setHoveredImage] = useState<string | null>(null);

  // Combine all images, most recent first
  const allImages = [...img2imgImages, ...generatedImages].reverse();

  const handleDownload = async (imagePath: string) => {
    try {
      const imageUrl = api.getImageUrl(imagePath);
      const response = await fetch(imageUrl);
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = imagePath.split('/').pop() || 'image.png';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Download failed:', error);
    }
  };

  const handleDelete = async (imagePath: string) => {
    try {
      // Call backend to delete
      await fetch(`http://localhost:8000/delete-image?path=${encodeURIComponent(imagePath)}`, {
        method: 'DELETE'
      });
      // Remove from store
      removeImage(imagePath);
    } catch (error) {
      console.error('Delete failed:', error);
      // Still remove from UI
      removeImage(imagePath);
    }
  };

  const handleSelect = (imagePath: string) => {
    if (onSelectImage) {
      onSelectImage(imagePath);
    }
  };

  const getCreativityColor = (creativity: number | undefined) => {
    if (creativity === undefined) return 'bg-zinc-600';
    // Purple (low) -> Blue (mid) -> Green (high)
    const hue = 260 - (creativity * 160);
    return `hsl(${hue}, 70%, 40%)`;
  };

  if (allImages.length === 0) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-zinc-500 p-8">
        <div className="w-16 h-16 mb-4 rounded-xl bg-zinc-800/50 flex items-center justify-center">
          <ZoomIn className="w-8 h-8" />
        </div>
        <p className="text-sm text-center">No images yet</p>
        <p className="text-xs text-zinc-600 mt-1">Generated images will appear here</p>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-zinc-800">
        <h2 className="text-sm font-medium text-zinc-300">
          Gallery ({allImages.length})
        </h2>
        <div className="flex items-center gap-2 text-xs text-zinc-500">
          <span className="flex items-center gap-1">
            <Heart className="w-3 h-3" /> {allImages.filter(i => i.liked).length}
          </span>
        </div>
      </div>

      {/* Grid */}
      <div className="flex-1 overflow-y-auto p-3">
        <div className="grid grid-cols-2 gap-2">
          {allImages.map((image, index) => (
            <div
              key={`${image.path}-${index}`}
              className={`relative group aspect-square rounded-lg overflow-hidden cursor-pointer
                ${selectedImage === image.path ? 'ring-2 ring-violet-500' : ''}
                ${image.liked ? 'ring-2 ring-pink-500/50' : ''}
              `}
              onMouseEnter={() => setHoveredImage(image.path)}
              onMouseLeave={() => setHoveredImage(null)}
              onClick={() => handleSelect(image.path)}
            >
              <img
                src={api.getImageUrl(image.path)}
                alt={`Generated ${index}`}
                className="w-full h-full object-cover"
                loading="lazy"
              />

              {/* Creativity badge */}
              {image.creativity !== undefined && (
                <div
                  className="absolute top-1.5 left-1.5 px-1.5 py-0.5 rounded text-[10px] font-medium text-white"
                  style={{ backgroundColor: getCreativityColor(image.creativity) }}
                >
                  {image.creativity.toFixed(1)}
                </div>
              )}

              {/* Liked indicator */}
              {image.liked && (
                <div className="absolute top-1.5 right-1.5">
                  <Heart className="w-4 h-4 text-pink-500 fill-pink-500" />
                </div>
              )}

              {/* Hover overlay with actions */}
              {hoveredImage === image.path && (
                <div className="absolute inset-0 bg-black/60 flex items-center justify-center gap-2 transition-opacity">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      likeImage(image.path);
                    }}
                    className={`p-2 rounded-full transition-colors ${
                      image.liked
                        ? 'bg-pink-500 text-white'
                        : 'bg-zinc-700 hover:bg-pink-500 text-zinc-300 hover:text-white'
                    }`}
                    title="Like"
                  >
                    <Heart className={`w-4 h-4 ${image.liked ? 'fill-white' : ''}`} />
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      setLightboxImage(image.path);
                    }}
                    className="p-2 rounded-full bg-zinc-700 hover:bg-zinc-600 text-zinc-300 hover:text-white transition-colors"
                    title="View full size"
                  >
                    <ZoomIn className="w-4 h-4" />
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDownload(image.path);
                    }}
                    className="p-2 rounded-full bg-zinc-700 hover:bg-emerald-500 text-zinc-300 hover:text-white transition-colors"
                    title="Download"
                  >
                    <Download className="w-4 h-4" />
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDelete(image.path);
                    }}
                    className="p-2 rounded-full bg-zinc-700 hover:bg-red-500 text-zinc-300 hover:text-white transition-colors"
                    title="Delete"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Lightbox */}
      {lightboxImage && (
        <div
          className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center p-8"
          onClick={() => setLightboxImage(null)}
        >
          <button
            className="absolute top-4 right-4 p-2 rounded-full bg-zinc-800 hover:bg-zinc-700 text-white"
            onClick={() => setLightboxImage(null)}
          >
            <X className="w-6 h-6" />
          </button>
          <img
            src={api.getImageUrl(lightboxImage)}
            alt="Full size"
            className="max-w-full max-h-full object-contain rounded-lg"
            onClick={(e) => e.stopPropagation()}
          />
          <div className="absolute bottom-4 left-1/2 -translate-x-1/2 flex gap-2">
            <button
              onClick={(e) => {
                e.stopPropagation();
                likeImage(lightboxImage);
              }}
              className="p-3 rounded-full bg-zinc-800 hover:bg-pink-500 text-white transition-colors"
            >
              <Heart className="w-5 h-5" />
            </button>
            <button
              onClick={(e) => {
                e.stopPropagation();
                handleDownload(lightboxImage);
              }}
              className="p-3 rounded-full bg-zinc-800 hover:bg-emerald-500 text-white transition-colors"
            >
              <Download className="w-5 h-5" />
            </button>
            <button
              onClick={(e) => {
                e.stopPropagation();
                handleDelete(lightboxImage);
                setLightboxImage(null);
              }}
              className="p-3 rounded-full bg-zinc-800 hover:bg-red-500 text-white transition-colors"
            >
              <Trash2 className="w-5 h-5" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
