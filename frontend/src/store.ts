import { create } from 'zustand';

export interface GeneratedImage {
  path: string;
  strength?: number;
  seed?: number;
  liked?: boolean;
}

export interface RefinedImage {
  original: string;
  refined: string;
}

export interface LoRA {
  name: string;
  path: string;
  valid?: boolean;
}

interface TrainingStatus {
  active: boolean;
  progress: number;
  message: string;
}

interface SetupCheck {
  status: 'ok' | 'warning' | 'error';
  message: string;
  [key: string]: unknown;
}

interface SetupStatus {
  active: boolean;
  ready: boolean;
  progress: number;
  message: string;
  checks: Record<string, SetupCheck>;
}

interface AppState {
  status: {
    training?: TrainingStatus;
    setup?: SetupStatus;
  };
  generatedImages: GeneratedImage[];
  refinedImages: RefinedImage[];
  loras: LoRA[];

  setStatus: (key: string, value: TrainingStatus | SetupStatus) => void;
  setSetupStatus: (status: SetupStatus) => void;
  addGeneratedImage: (image: GeneratedImage) => void;
  clearGeneratedImages: () => void;
  likeImage: (path: string) => void;
  addRefinedImage: (image: RefinedImage) => void;
  setLoras: (loras: LoRA[]) => void;
}

export const useStore = create<AppState>((set) => ({
  status: {},
  generatedImages: [],
  refinedImages: [],
  loras: [],

  setStatus: (key, value) => set((state) => ({
    status: { ...state.status, [key]: value }
  })),

  setSetupStatus: (status) => set((state) => ({
    status: { ...state.status, setup: status }
  })),

  addGeneratedImage: (image) => set((state) => ({
    generatedImages: [...state.generatedImages, image]
  })),

  clearGeneratedImages: () => set({ generatedImages: [] }),

  likeImage: (path) => set((state) => ({
    generatedImages: state.generatedImages.map(img =>
      img.path === path ? { ...img, liked: true } : img
    )
  })),

  addRefinedImage: (image) => set((state) => ({
    refinedImages: [...state.refinedImages, image]
  })),

  setLoras: (loras) => set({ loras })
}));
