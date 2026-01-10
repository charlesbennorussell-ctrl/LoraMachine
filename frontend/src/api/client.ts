const API_BASE = 'http://localhost:8000';

interface TrainingConfig {
  name: string;
  steps: number;
  learning_rate: number;
  batch_size: number;
  resolution: number;
  trigger_word: string;
}

interface GenerationConfig {
  prompt: string;
  negative_prompt?: string;
  lora_path: string;
  lora_strength: number;
  steps: number;
  guidance_scale: number;
  width: number;
  height: number;
  seed?: number | null;
}

interface IterationConfig extends Omit<GenerationConfig, 'lora_strength'> {
  strength_start: number;
  strength_end: number;
  strength_step: number;
}

export const api = {
  async getStatus() {
    const res = await fetch(`${API_BASE}/status`);
    return res.json();
  },

  async uploadTrainingImages(formData: FormData) {
    const res = await fetch(`${API_BASE}/upload-training-images`, {
      method: 'POST',
      body: formData
    });
    return res.json();
  },

  async getTrainingImages() {
    const res = await fetch(`${API_BASE}/training-images`);
    return res.json();
  },

  async clearTrainingImages() {
    const res = await fetch(`${API_BASE}/training-images`, {
      method: 'DELETE'
    });
    return res.json();
  },

  async startTraining(config: TrainingConfig) {
    const res = await fetch(`${API_BASE}/train`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    });
    return res.json();
  },

  async generate(config: GenerationConfig) {
    const res = await fetch(`${API_BASE}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    });
    return res.json();
  },

  async runIteration(config: IterationConfig) {
    const res = await fetch(`${API_BASE}/iterate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    });
    return res.json();
  },

  async likeImage(imageId: string) {
    const res = await fetch(`${API_BASE}/like/${imageId}`, {
      method: 'POST'
    });
    return res.json();
  },

  async getLoras() {
    const res = await fetch(`${API_BASE}/loras`);
    return res.json();
  },

  async getGeneratedImages() {
    const res = await fetch(`${API_BASE}/generated-images`);
    return res.json();
  },

  async getRefinedImages() {
    const res = await fetch(`${API_BASE}/refined-images`);
    return res.json();
  },

  // Helper to get full image URL
  getImageUrl(path: string) {
    if (path.startsWith('http')) return path;
    return `${API_BASE}${path}`;
  }
};
