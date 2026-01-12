import { useState, useCallback } from 'react';
import { Trash2, Edit3, Upload, X, Check, Image as ImageIcon, AlertTriangle } from 'lucide-react';
import { useDropzone } from 'react-dropzone';
import { api } from '../api/client';
import { useStore, LoRA } from '../store';

interface LoRACardProps {
  lora: LoRA;
}

function LoRACard({ lora }: LoRACardProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [editForm, setEditForm] = useState({
    trigger_word: lora.trigger_word || 'ohwx',
    description: lora.description || ''
  });
  const { removeLora, updateLora } = useStore();

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) return;

    setIsUploading(true);
    try {
      const result = await api.uploadLoraThumbnail(lora.name, acceptedFiles[0]);
      if (result.thumbnail) {
        updateLora(lora.name, { thumbnail: result.thumbnail });
      }
    } catch (error) {
      console.error('Failed to upload thumbnail:', error);
    }
    setIsUploading(false);
  }, [lora.name, updateLora]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'image/*': ['.png', '.jpg', '.jpeg', '.webp']
    },
    maxFiles: 1
  });

  const handleSave = async () => {
    try {
      await api.updateLora(lora.name, editForm);
      updateLora(lora.name, editForm);
      setIsEditing(false);
    } catch (error) {
      console.error('Failed to update LoRA:', error);
    }
  };

  const handleDelete = async () => {
    try {
      await api.deleteLora(lora.name);
      removeLora(lora.name);
    } catch (error) {
      console.error('Failed to delete LoRA:', error);
    }
  };

  return (
    <div className="bg-zinc-900 rounded-lg border border-zinc-800 overflow-hidden">
      {/* Thumbnail Area */}
      <div className="relative aspect-square bg-zinc-800">
        {lora.thumbnail ? (
          <img
            src={api.getImageUrl(lora.thumbnail)}
            alt={lora.name}
            className="w-full h-full object-cover"
          />
        ) : (
          <div
            {...getRootProps()}
            className={`w-full h-full flex flex-col items-center justify-center cursor-pointer transition-colors
              ${isDragActive ? 'bg-violet-500/20' : 'hover:bg-zinc-700'}`}
          >
            <input {...getInputProps()} />
            {isUploading ? (
              <div className="text-center">
                <div className="animate-spin w-8 h-8 border-2 border-violet-500 border-t-transparent rounded-full mx-auto" />
                <p className="text-xs text-zinc-500 mt-2">Uploading...</p>
              </div>
            ) : (
              <>
                <ImageIcon className="w-8 h-8 text-zinc-600" />
                <p className="text-xs text-zinc-500 mt-2">Click to add thumbnail</p>
              </>
            )}
          </div>
        )}

        {/* Validity badge */}
        {!lora.valid && (
          <div className="absolute top-2 right-2 p-1 bg-amber-500/90 rounded" title="Missing config files">
            <AlertTriangle className="w-4 h-4 text-white" />
          </div>
        )}

        {/* Thumbnail change overlay when thumbnail exists */}
        {lora.thumbnail && (
          <div
            {...getRootProps()}
            className="absolute inset-0 bg-black/60 opacity-0 hover:opacity-100 transition-opacity flex items-center justify-center cursor-pointer"
          >
            <input {...getInputProps()} />
            <div className="text-center">
              <Upload className="w-6 h-6 mx-auto" />
              <p className="text-xs mt-1">Change thumbnail</p>
            </div>
          </div>
        )}
      </div>

      {/* Info */}
      <div className="p-3">
        <h3 className="font-medium text-sm truncate" title={lora.name}>
          {lora.name}
        </h3>

        {isEditing ? (
          <div className="mt-2 space-y-2">
            <div>
              <label className="text-xs text-zinc-500">Trigger Word</label>
              <input
                type="text"
                value={editForm.trigger_word}
                onChange={e => setEditForm(f => ({ ...f, trigger_word: e.target.value }))}
                className="w-full bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-xs
                  focus:outline-none focus:ring-1 focus:ring-violet-500"
              />
            </div>
            <div>
              <label className="text-xs text-zinc-500">Description</label>
              <textarea
                value={editForm.description}
                onChange={e => setEditForm(f => ({ ...f, description: e.target.value }))}
                rows={2}
                className="w-full bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-xs resize-none
                  focus:outline-none focus:ring-1 focus:ring-violet-500"
              />
            </div>
            <div className="flex gap-2">
              <button
                onClick={handleSave}
                className="flex-1 flex items-center justify-center gap-1 px-2 py-1 bg-emerald-600 hover:bg-emerald-500 rounded text-xs"
              >
                <Check className="w-3 h-3" />
                Save
              </button>
              <button
                onClick={() => setIsEditing(false)}
                className="flex-1 flex items-center justify-center gap-1 px-2 py-1 bg-zinc-700 hover:bg-zinc-600 rounded text-xs"
              >
                <X className="w-3 h-3" />
                Cancel
              </button>
            </div>
          </div>
        ) : (
          <>
            <p className="text-xs text-violet-400 mt-1">
              Trigger: <span className="font-mono">{lora.trigger_word || 'ohwx'}</span>
            </p>
            {lora.description && (
              <p className="text-xs text-zinc-500 mt-1 line-clamp-2">{lora.description}</p>
            )}

            {/* Actions */}
            <div className="flex gap-2 mt-3">
              <button
                onClick={() => setIsEditing(true)}
                className="flex-1 flex items-center justify-center gap-1 px-2 py-1.5 bg-zinc-800 hover:bg-zinc-700 rounded text-xs transition-colors"
              >
                <Edit3 className="w-3 h-3" />
                Edit
              </button>
              {isDeleting ? (
                <div className="flex-1 flex gap-1">
                  <button
                    onClick={handleDelete}
                    className="flex-1 px-2 py-1.5 bg-red-600 hover:bg-red-500 rounded text-xs"
                  >
                    Confirm
                  </button>
                  <button
                    onClick={() => setIsDeleting(false)}
                    className="flex-1 px-2 py-1.5 bg-zinc-700 hover:bg-zinc-600 rounded text-xs"
                  >
                    Cancel
                  </button>
                </div>
              ) : (
                <button
                  onClick={() => setIsDeleting(true)}
                  className="flex-1 flex items-center justify-center gap-1 px-2 py-1.5 bg-zinc-800 hover:bg-red-600 rounded text-xs transition-colors"
                >
                  <Trash2 className="w-3 h-3" />
                  Delete
                </button>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export function LoRAManager() {
  const { loras, setLoras } = useStore();

  const refreshLoras = async () => {
    try {
      const data = await api.getLoras();
      setLoras(data.loras);
    } catch (error) {
      console.error('Failed to refresh LoRAs:', error);
    }
  };

  if (loras.length === 0) {
    return (
      <div className="text-center py-12">
        <ImageIcon className="w-12 h-12 mx-auto text-zinc-600" />
        <h3 className="text-lg font-medium mt-4">No LoRAs Found</h3>
        <p className="text-zinc-500 mt-1">
          Train your first LoRA in the Train tab to get started.
        </p>
      </div>
    );
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-lg font-medium">Manage LoRAs</h2>
          <p className="text-sm text-zinc-400">
            Edit metadata, add thumbnails, or delete LoRAs
          </p>
        </div>
        <button
          onClick={refreshLoras}
          className="px-3 py-1.5 bg-zinc-800 hover:bg-zinc-700 rounded-lg text-sm transition-colors"
        >
          Refresh
        </button>
      </div>

      <div className="grid grid-cols-4 gap-4">
        {loras.map(lora => (
          <LoRACard key={lora.name} lora={lora} />
        ))}
      </div>
    </div>
  );
}
