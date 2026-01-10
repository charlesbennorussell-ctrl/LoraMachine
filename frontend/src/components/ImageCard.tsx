import { Heart, Download, ZoomIn } from 'lucide-react';
import { useState } from 'react';
import { api } from '../api/client';

interface ImageCardProps {
  path: string;
  strength?: number;
  liked?: boolean;
  onLike?: () => void;
  showStrength?: boolean;
}

export function ImageCard({ path, strength, liked, onLike, showStrength = true }: ImageCardProps) {
  const [isLoading, setIsLoading] = useState(true);
  const [showModal, setShowModal] = useState(false);

  const imageUrl = api.getImageUrl(path);

  return (
    <>
      <div className="group relative rounded-lg overflow-hidden bg-zinc-900">
        {/* Loading skeleton */}
        {isLoading && (
          <div className="absolute inset-0 loading-shimmer" />
        )}

        {/* Image */}
        <img
          src={imageUrl}
          alt={`Generated image ${strength ? `at strength ${strength}` : ''}`}
          className={`w-full aspect-square object-cover transition-opacity duration-300 ${
            isLoading ? 'opacity-0' : 'opacity-100'
          }`}
          onLoad={() => setIsLoading(false)}
          onError={() => setIsLoading(false)}
        />

        {/* Strength badge */}
        {showStrength && strength !== undefined && (
          <div
            className="absolute top-2 left-2 px-2 py-0.5 rounded text-xs font-medium"
            style={{
              backgroundColor: `hsl(${120 - strength * 120}, 70%, 40%)`,
              color: 'white'
            }}
          >
            {strength.toFixed(1)}
          </div>
        )}

        {/* Liked indicator */}
        {liked && (
          <div className="absolute top-2 right-2">
            <Heart className="w-5 h-5 text-red-500" fill="currentColor" />
          </div>
        )}

        {/* Hover overlay */}
        <div className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-transparent
          opacity-0 group-hover:opacity-100 transition-opacity">
          <div className="absolute bottom-0 left-0 right-0 p-3 flex items-center justify-between">
            <button
              onClick={() => setShowModal(true)}
              className="p-1.5 rounded-full bg-white/20 hover:bg-white/30 transition-colors"
              title="View full size"
            >
              <ZoomIn className="w-4 h-4" />
            </button>

            <div className="flex gap-2">
              {onLike && (
                <button
                  onClick={onLike}
                  className={`p-1.5 rounded-full transition-colors
                    ${liked
                      ? 'bg-red-500 text-white'
                      : 'bg-white/20 hover:bg-red-500 hover:text-white'
                    }`}
                  title="Like and refine"
                >
                  <Heart className="w-4 h-4" fill={liked ? 'currentColor' : 'none'} />
                </button>
              )}
              <a
                href={imageUrl}
                download
                className="p-1.5 rounded-full bg-white/20 hover:bg-white/30 transition-colors"
                title="Download"
              >
                <Download className="w-4 h-4" />
              </a>
            </div>
          </div>
        </div>
      </div>

      {/* Modal */}
      {showModal && (
        <div
          className="fixed inset-0 bg-black/90 z-50 flex items-center justify-center p-8"
          onClick={() => setShowModal(false)}
        >
          <div className="relative max-w-5xl max-h-full">
            <img
              src={imageUrl}
              alt="Full size"
              className="max-w-full max-h-[90vh] object-contain rounded-lg"
            />
            {showStrength && strength !== undefined && (
              <div className="absolute top-4 left-4 px-3 py-1 bg-black/60 rounded-lg">
                <span className="text-sm">LoRA Strength: {strength.toFixed(2)}</span>
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
}
