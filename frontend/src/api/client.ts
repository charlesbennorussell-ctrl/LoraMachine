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

// Img2Img configs
interface Img2ImgGenerationConfig {
  prompt: string;
  negative_prompt?: string;
  lora_path: string;
  lora_strength: number;  // Usually 1.0 for full LoRA
  creativity: number;     // 0-1, how much to deviate from input
  steps: number;
  guidance_scale: number;
  width: number;
  height: number;
  seed?: number | null;
}

interface Img2ImgIterationConfig {
  prompt: string;
  negative_prompt?: string;
  lora_path: string;
  lora_strength: number;
  steps: number;
  guidance_scale: number;
  width: number;
  height: number;
  seed?: number | null;
  creativity_values?: number[];  // Default: [0.0, 0.2, 0.4, 0.6, 0.8]
}

interface LoRAUpdateConfig {
  name?: string;
  trigger_word?: string;
  description?: string;
}

export const api = {
  async getStatus() {
    const res = await fetch(`${API_BASE}/status`);
    return res.json();
  },

  async getSetupStatus() {
    const res = await fetch(`${API_BASE}/setup/status`);
    return res.json();
  },

  async runSetup() {
    const res = await fetch(`${API_BASE}/setup`, {
      method: 'POST'
    });
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

  async getLoraDetails(loraName: string) {
    const res = await fetch(`${API_BASE}/loras/${encodeURIComponent(loraName)}`);
    return res.json();
  },

  async updateLora(loraName: string, config: LoRAUpdateConfig) {
    const res = await fetch(`${API_BASE}/loras/${encodeURIComponent(loraName)}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    });
    return res.json();
  },

  async deleteLora(loraName: string) {
    const res = await fetch(`${API_BASE}/loras/${encodeURIComponent(loraName)}`, {
      method: 'DELETE'
    });
    return res.json();
  },

  async uploadLoraThumbnail(loraName: string, file: File) {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${API_BASE}/loras/${encodeURIComponent(loraName)}/thumbnail`, {
      method: 'POST',
      body: formData
    });
    return res.json();
  },

  // ===== IMG2IMG APIs =====

  async uploadInputImage(file: File) {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${API_BASE}/upload-input-image`, {
      method: 'POST',
      body: formData
    });
    return res.json();
  },

  async generateImg2Img(config: Img2ImgGenerationConfig, inputImagePath: string) {
    const res = await fetch(`${API_BASE}/generate-img2img?input_image_path=${encodeURIComponent(inputImagePath)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    });
    return res.json();
  },

  async runImg2ImgIteration(config: Img2ImgIterationConfig, inputImagePath: string) {
    const res = await fetch(`${API_BASE}/iterate-img2img?input_image_path=${encodeURIComponent(inputImagePath)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config)
    });
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
