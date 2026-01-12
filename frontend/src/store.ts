import { create } from 'zustand';

export interface GeneratedImage {
  path: string;
  strength?: number;  // For LoRA strength iterations
  creativity?: number;  // For img2img creativity iterations
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
  thumbnail?: string | null;
  trigger_word?: string;
  description?: string;
  created_at?: string;
}

export interface InputImage {
  path: string;
  url: string;
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
  img2imgImages: GeneratedImage[];  // Separate array for img2img results
  refinedImages: RefinedImage[];
  loras: LoRA[];
  inputImage: InputImage | null;  // Current input image for img2img

  setStatus: (key: string, value: TrainingStatus | SetupStatus) => void;
  setSetupStatus: (status: SetupStatus) => void;
  addGeneratedImage: (image: GeneratedImage) => void;
  clearGeneratedImages: () => void;
  addImg2ImgImage: (image: GeneratedImage) => void;
  clearImg2ImgImages: () => void;
  likeImage: (path: string) => void;
  removeImage: (path: string) => void;
  addRefinedImage: (image: RefinedImage) => void;
  setLoras: (loras: LoRA[]) => void;
  updateLora: (name: string, updates: Partial<LoRA>) => void;
  removeLora: (name: string) => void;
  setInputImage: (image: InputImage | null) => void;
}

export const useStore = create<AppState>((set) => ({
  status: {},
  generatedImages: [],
  img2imgImages: [],
  refinedImages: [],
  loras: [],
  inputImage: null,

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

  addImg2ImgImage: (image) => set((state) => ({
    img2imgImages: [...state.img2imgImages, image]
  })),

  clearImg2ImgImages: () => set({ img2imgImages: [] }),

  likeImage: (path) => set((state) => ({
    generatedImages: state.generatedImages.map(img =>
      img.path === path ? { ...img, liked: !img.liked } : img
    ),
    img2imgImages: state.img2imgImages.map(img =>
      img.path === path ? { ...img, liked: !img.liked } : img
    )
  })),

  removeImage: (path) => set((state) => ({
    generatedImages: state.generatedImages.filter(img => img.path !== path),
    img2imgImages: state.img2imgImages.filter(img => img.path !== path)
  })),

  addRefinedImage: (image) => set((state) => ({
    refinedImages: [...state.refinedImages, image]
  })),

  setLoras: (loras) => set({ loras }),

  updateLora: (name, updates) => set((state) => ({
    loras: state.loras.map(lora =>
      lora.name === name ? { ...lora, ...updates } : lora
    )
  })),

  removeLora: (name) => set((state) => ({
    loras: state.loras.filter(lora => lora.name !== name)
  })),

  setInputImage: (image) => set({ inputImage: image })
}));
