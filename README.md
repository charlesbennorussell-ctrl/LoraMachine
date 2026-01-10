# Flux LoRA Training & Iteration Pipeline

A complete local Flux LoRA training and automated iteration pipeline with a React web interface. Train custom LoRAs, generate images at varying strengths (0.1 to 1.0), and auto-refine your best results.

**Target Hardware:** NVIDIA RTX 4080 (16GB VRAM)

## Features

- **LoRA Training**: Train custom LoRAs on Flux using your own images
- **Iteration Sweep**: Generate images at LoRA strengths from 0.1 to 1.0
- **Like & Refine**: Mark favorite images for automatic enhancement
- **Multiple Refinement Options**:
  - Real-ESRGAN upscaling
  - Qwen2-VL analysis + Flux Fill (optional)
  - Flux img2img refinement

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- NVIDIA GPU with 16GB+ VRAM
- CUDA 12.1+

### Installation

1. **Run Setup Script**:
   ```bash
   # Windows
   scripts\setup.bat

   # Linux/Mac
   chmod +x scripts/setup.sh
   ./scripts/setup.sh
   ```

2. **Accept Flux License**:
   Visit https://huggingface.co/black-forest-labs/FLUX.1-dev and accept the license.

3. **Login to HuggingFace**:
   ```bash
   huggingface-cli login
   ```

4. **Start the Application**:
   ```bash
   # Terminal 1 - Backend
   cd backend
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   python main.py

   # Terminal 2 - Frontend
   cd frontend
   npm run dev
   ```

5. **Open Browser**: http://localhost:5173

## Usage

### 1. Training

1. Go to the "Train" tab
2. Upload 10-20 high-quality images of your subject
3. Set a name and trigger word (e.g., "ohwx person")
4. Configure training steps (1000-2000 recommended)
5. Click "Start Training"

### 2. Generation

1. Go to the "Generate" tab
2. Select your trained LoRA
3. Write a prompt including your trigger word
4. Adjust LoRA strength with the slider
5. Click "Generate"

### 3. Iteration

1. Go to the "Iterate" tab
2. Select your LoRA and write a prompt
3. Click "Run Full Iteration"
4. The system generates images at 0.1, 0.2, ... 1.0 strength
5. Click the heart icon on your favorite images

### 4. Refinement

Liked images are automatically refined and appear in the "Refined" tab.

## Project Structure

```
flux-lora-pipeline/
├── backend/
│   ├── main.py              # FastAPI server
│   ├── training/            # LoRA training code
│   ├── inference/           # Image generation
│   └── refinement/          # Image enhancement
├── frontend/
│   └── src/                 # React UI
├── models/
│   ├── flux/               # Flux base model (auto-downloaded)
│   ├── loras/              # Your trained LoRAs
│   └── refiners/           # Enhancement models
├── training_data/          # Upload training images here
├── outputs/
│   ├── generated/          # Generated images
│   ├── liked/              # Favorited images
│   └── refined/            # Enhanced images
└── configs/
    └── default_training.yaml
```

## API Endpoints

- `POST /train` - Start LoRA training
- `POST /generate` - Generate single image
- `POST /iterate` - Run strength iteration sweep
- `POST /like/{image_id}` - Like and queue for refinement
- `GET /loras` - List trained LoRAs
- `GET /status` - System status
- `WS /ws` - WebSocket for real-time updates

## Memory Optimization

The pipeline is optimized for 16GB VRAM:

- CPU offloading for model components
- VAE slicing and tiling
- bfloat16 precision
- xformers attention (when available)

## Troubleshooting

### CUDA Out of Memory
- Reduce resolution to 768 or 512
- Reduce batch size to 1
- Close other GPU applications

### Flux Download Fails
- Ensure you've accepted the license on HuggingFace
- Run `huggingface-cli login` with a valid token

### Training Loss Not Decreasing
- Try lower learning rate (5e-5)
- Add more training images
- Use longer captions

## License

This project is for personal use. The Flux model has its own license terms at black-forest-labs/FLUX.1-dev.
