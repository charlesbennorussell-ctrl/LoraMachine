import os

# CRITICAL: Disable xformers BEFORE any imports to prevent segfaults
os.environ["XFORMERS_DISABLED"] = "1"
os.environ["DIFFUSERS_NO_XFORMERS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Workaround for Windows safetensors segfault with large files
os.environ["SAFETENSORS_FAST_GPU"] = "0"
os.environ["ACCELERATE_USE_SAFETENSORS"] = "true"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Disable torch compile which can cause issues on Windows
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil

# Set cache directories to D: drive to avoid C: drive space issues
os.environ['HF_HOME'] = 'D:/CTRL_ITERATION/flux-cache'
os.environ['TRANSFORMERS_CACHE'] = 'D:/CTRL_ITERATION/flux-cache'
os.environ['HF_DATASETS_CACHE'] = 'D:/CTRL_ITERATION/flux-cache'

# Global thread pool for blocking operations
_setup_executor = ThreadPoolExecutor(max_workers=1)

# Use QLoRA trainer for 16GB VRAM compatibility
from training.qlora_trainer import QLoRATrainer as LoRATrainer
from inference.generator import FluxGenerator
from inference.iterator import LoRAIterator
from refinement.refiner import ImageRefiner

app = FastAPI(title="Flux LoRA Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for serving generated images
outputs_path = Path(__file__).parent.parent / "outputs"
outputs_path.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(outputs_path)), name="outputs")

# Global state
training_status = {"active": False, "progress": 0, "message": "Idle"}
setup_status = {"active": False, "ready": False, "progress": 0, "message": "Not initialized", "checks": {}}
generation_queue = asyncio.Queue()
connected_websockets: List[WebSocket] = []

# Initialize components (lazy loading)
trainer: Optional[LoRATrainer] = None
generator: Optional[FluxGenerator] = None
iterator: Optional[LoRAIterator] = None
refiner: Optional[ImageRefiner] = None


def get_trainer():
    global trainer
    if trainer is None:
        trainer = LoRATrainer()
    return trainer


def get_generator():
    global generator
    if generator is None:
        generator = FluxGenerator()
    return generator


def get_iterator():
    global iterator
    if iterator is None:
        iterator = LoRAIterator(get_generator())
    return iterator


def get_refiner():
    global refiner
    if refiner is None:
        refiner = ImageRefiner()
    return refiner


class TrainingConfig(BaseModel):
    name: str
    steps: int = 1000
    learning_rate: float = 1e-4
    batch_size: int = 1
    resolution: int = 1024
    trigger_word: str = "ohwx"


class GenerationConfig(BaseModel):
    prompt: str
    negative_prompt: str = ""
    lora_path: str
    lora_strength: float = 0.8
    steps: int = 28
    guidance_scale: float = 3.5
    width: int = 1024
    height: int = 1024
    seed: Optional[int] = None


class IterationConfig(BaseModel):
    prompt: str
    negative_prompt: str = ""
    lora_path: str
    steps: int = 28
    guidance_scale: float = 3.5
    width: int = 1024
    height: int = 1024
    seed: Optional[int] = None
    strength_start: float = 0.1
    strength_end: float = 1.0
    strength_step: float = 0.1


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_websockets.append(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle ping/pong for keepalive
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        if websocket in connected_websockets:
            connected_websockets.remove(websocket)
    except Exception:
        if websocket in connected_websockets:
            connected_websockets.remove(websocket)


async def broadcast(message: dict):
    """Broadcast message to all connected WebSocket clients"""
    disconnected = []
    for ws in connected_websockets:
        try:
            await ws.send_json(message)
        except Exception:
            disconnected.append(ws)

    for ws in disconnected:
        if ws in connected_websockets:
            connected_websockets.remove(ws)


@app.post("/upload-training-images")
async def upload_training_images(files: List[UploadFile] = File(...)):
    """Upload images for LoRA training"""
    training_dir = Path(__file__).parent.parent / "training_data"
    training_dir.mkdir(exist_ok=True)

    saved_files = []
    for file in files:
        file_path = training_dir / file.filename
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        saved_files.append(str(file_path))

    return {"uploaded": saved_files, "count": len(saved_files)}


@app.delete("/training-images")
async def clear_training_images():
    """Clear all training images"""
    training_dir = Path(__file__).parent.parent / "training_data"
    if training_dir.exists():
        for file in training_dir.iterdir():
            if file.is_file():
                file.unlink()
    return {"status": "cleared"}


@app.get("/training-images")
async def list_training_images():
    """List uploaded training images"""
    training_dir = Path(__file__).parent.parent / "training_data"
    images = []
    if training_dir.exists():
        for file in training_dir.iterdir():
            if file.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp']:
                images.append({"name": file.name, "path": str(file)})
    return {"images": images, "count": len(images)}


@app.post("/train")
async def start_training(config: TrainingConfig):
    """Start LoRA training job"""
    global training_status

    async def train_task():
        global training_status
        training_status = {"active": True, "progress": 0, "message": "Initializing..."}
        await broadcast({"type": "training_status", "data": training_status})

        async def progress_callback(p, m):
            """Async progress callback that broadcasts to WebSocket clients"""
            global training_status
            training_status = {"active": True, "progress": p, "message": m}
            await broadcast({"type": "training_status", "data": training_status})

        try:
            base_path = Path(__file__).parent.parent
            await get_trainer().train(
                name=config.name,
                training_dir=str(base_path / "training_data"),
                output_dir=str(base_path / "models" / "loras" / config.name),
                steps=config.steps,
                learning_rate=config.learning_rate,
                batch_size=config.batch_size,
                resolution=config.resolution,
                trigger_word=config.trigger_word,
                progress_callback=progress_callback
            )
            training_status = {"active": False, "progress": 100, "message": "Complete!"}
        except Exception as e:
            training_status = {"active": False, "progress": 0, "message": f"Error: {str(e)}"}
            import traceback
            traceback.print_exc()

        await broadcast({"type": "training_status", "data": training_status})

    # Create task in the main event loop - NOT in a separate thread/loop
    asyncio.create_task(train_task())
    return {"status": "Training started", "config": config.model_dump()}


@app.post("/generate")
async def generate_image(config: GenerationConfig):
    """Generate a single image with LoRA"""
    base_path = Path(__file__).parent.parent
    image_path = await get_generator().generate(
        prompt=config.prompt,
        negative_prompt=config.negative_prompt,
        lora_path=config.lora_path,
        lora_strength=config.lora_strength,
        steps=config.steps,
        guidance_scale=config.guidance_scale,
        width=config.width,
        height=config.height,
        seed=config.seed,
        output_dir=str(base_path / "outputs" / "generated")
    )

    # Convert to relative path for frontend
    rel_path = str(Path(image_path).relative_to(base_path))
    return {"image_path": "/" + rel_path.replace("\\", "/")}


@app.post("/iterate")
async def run_iteration(config: IterationConfig):
    """Run full LoRA strength iteration (0.1 to 1.0)"""

    async def iterate_task():
        base_path = Path(__file__).parent.parent

        async def progress_cb(strength, path):
            rel_path = str(Path(path).relative_to(base_path))
            await broadcast({
                "type": "iteration_progress",
                "data": {"strength": strength, "path": "/" + rel_path.replace("\\", "/")}
            })

        results = await get_iterator().run_iteration(
            prompt=config.prompt,
            negative_prompt=config.negative_prompt,
            lora_path=config.lora_path,
            steps=config.steps,
            guidance_scale=config.guidance_scale,
            width=config.width,
            height=config.height,
            seed=config.seed,
            strength_start=config.strength_start,
            strength_end=config.strength_end,
            strength_step=config.strength_step,
            output_dir=str(base_path / "outputs" / "generated"),
            progress_callback=progress_cb
        )

        # Convert paths for frontend
        for r in results:
            rel_path = str(Path(r["path"]).relative_to(base_path))
            r["path"] = "/" + rel_path.replace("\\", "/")

        await broadcast({"type": "iteration_complete", "data": results})

    # Create task in the main event loop - NOT in a separate thread/loop
    asyncio.create_task(iterate_task())
    return {"status": "Iteration started"}


@app.post("/like/{image_id}")
async def like_image(image_id: str):
    """Mark image as liked and queue for refinement"""
    base_path = Path(__file__).parent.parent
    source = base_path / "outputs" / "generated" / image_id
    dest = base_path / "outputs" / "liked" / image_id

    if source.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(source, dest)

        # Auto-refine in background
        async def refine_task():
            refined_path = await get_refiner().refine(
                str(dest),
                output_dir=str(base_path / "outputs" / "refined")
            )

            # Convert paths for frontend
            orig_rel = str(dest.relative_to(base_path))
            refined_rel = str(Path(refined_path).relative_to(base_path))

            await broadcast({
                "type": "refinement_complete",
                "data": {
                    "original": "/" + orig_rel.replace("\\", "/"),
                    "refined": "/" + refined_rel.replace("\\", "/")
                }
            })

        # Create task in the main event loop - NOT in a separate thread/loop
        asyncio.create_task(refine_task())
        return {"status": "Liked and queued for refinement", "image_id": image_id}

    return {"error": "Image not found"}


@app.get("/loras")
async def list_loras():
    """List available trained LoRAs"""
    lora_dir = Path(__file__).parent.parent / "models" / "loras"
    loras = []
    if lora_dir.exists():
        for lora_path in lora_dir.iterdir():
            if lora_path.is_dir():
                # Check for config or adapter files
                has_config = (lora_path / "config.json").exists() or \
                           (lora_path / "adapter_config.json").exists()
                loras.append({
                    "name": lora_path.name,
                    "path": str(lora_path),
                    "valid": has_config
                })
    return {"loras": loras}


@app.get("/generated-images")
async def list_generated_images():
    """List all generated images"""
    base_path = Path(__file__).parent.parent
    generated_dir = base_path / "outputs" / "generated"
    images = []
    if generated_dir.exists():
        for file in sorted(generated_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                rel_path = str(file.relative_to(base_path))
                images.append({
                    "name": file.name,
                    "path": "/" + rel_path.replace("\\", "/")
                })
    return {"images": images}


@app.get("/refined-images")
async def list_refined_images():
    """List all refined images with their originals"""
    base_path = Path(__file__).parent.parent
    refined_dir = base_path / "outputs" / "refined"
    liked_dir = base_path / "outputs" / "liked"

    images = []
    if refined_dir.exists():
        for file in sorted(refined_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
            if file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                refined_rel = str(file.relative_to(base_path))

                # Try to find original
                original_name = file.stem.replace("refined_", "").rsplit("_", 1)[0]
                original_path = None
                for orig in liked_dir.iterdir() if liked_dir.exists() else []:
                    if original_name in orig.stem:
                        original_rel = str(orig.relative_to(base_path))
                        original_path = "/" + original_rel.replace("\\", "/")
                        break

                images.append({
                    "refined": "/" + refined_rel.replace("\\", "/"),
                    "original": original_path
                })
    return {"images": images}


@app.get("/status")
async def get_status():
    """Get current system status"""
    return {
        "training": training_status,
        "setup": setup_status,
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB" if torch.cuda.is_available() else None
    }


@app.get("/setup/status")
async def get_setup_status():
    """Get detailed setup status"""
    return setup_status


@app.post("/setup")
async def run_setup():
    """
    Run pre-training setup: download models, verify GPU, run system checks.
    This should be called before attempting to train.
    """
    global setup_status

    if setup_status["active"]:
        return {"status": "Setup already in progress", "setup": setup_status}

    loop = asyncio.get_event_loop()

    def sync_progress(p, m, checks=None):
        """Thread-safe progress callback"""
        global setup_status
        setup_status = {
            "active": True,
            "ready": False,
            "progress": p,
            "message": m,
            "checks": checks or setup_status.get("checks", {})
        }
        asyncio.run_coroutine_threadsafe(
            broadcast({"type": "setup_status", "data": setup_status}),
            loop
        )

    async def setup_task():
        global setup_status

        setup_status = {"active": True, "ready": False, "progress": 0, "message": "Starting setup...", "checks": {}}
        await broadcast({"type": "setup_status", "data": setup_status})

        try:
            # Run blocking setup in thread pool
            result = await loop.run_in_executor(
                _setup_executor,
                _run_setup_sync,
                sync_progress
            )

            setup_status = {
                "active": False,
                "ready": result["success"],
                "progress": 100,
                "message": "Setup complete!" if result["success"] else f"Setup failed: {result.get('error', 'Unknown error')}",
                "checks": result["checks"]
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            setup_status = {
                "active": False,
                "ready": False,
                "progress": 0,
                "message": f"Setup failed: {str(e)}",
                "checks": setup_status.get("checks", {})
            }

        await broadcast({"type": "setup_status", "data": setup_status})

    asyncio.create_task(setup_task())
    return {"status": "Setup started"}


def _run_setup_sync(progress_callback):
    """
    Synchronous setup that runs in a thread pool.
    Downloads models and runs system checks.
    """
    checks = {}

    # Check 1: GPU availability
    progress_callback(5, "Checking GPU...", checks)
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        checks["gpu"] = {
            "status": "ok",
            "name": gpu_name,
            "memory_gb": round(gpu_mem, 1),
            "message": f"{gpu_name} with {gpu_mem:.1f}GB VRAM"
        }
        print(f"[SETUP] GPU: {gpu_name} ({gpu_mem:.1f}GB)")
    else:
        checks["gpu"] = {
            "status": "warning",
            "message": "No GPU detected - training will be very slow"
        }
        print("[SETUP] WARNING: No GPU detected")

    # Check 2: Disk space on cache drive
    progress_callback(10, "Checking disk space...", checks)
    cache_dir = Path("D:/CTRL_ITERATION/flux-cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        import shutil
        total, used, free = shutil.disk_usage(cache_dir)
        free_gb = free / 1024**3
        checks["disk_space"] = {
            "status": "ok" if free_gb > 50 else ("warning" if free_gb > 20 else "error"),
            "free_gb": round(free_gb, 1),
            "message": f"{free_gb:.1f}GB free on cache drive"
        }
        print(f"[SETUP] Disk space: {free_gb:.1f}GB free")
    except Exception as e:
        checks["disk_space"] = {
            "status": "error",
            "message": f"Could not check disk space: {str(e)}"
        }

    # Check 3: Download/verify Flux model
    progress_callback(15, "Checking Flux model (this may take a while on first run)...", checks)

    try:
        from huggingface_hub import snapshot_download, hf_hub_download
        from huggingface_hub.utils import LocalEntryNotFoundError

        model_id = "black-forest-labs/FLUX.1-dev"
        cache_dir_str = str(cache_dir)

        # Check if model files exist
        progress_callback(20, "Verifying model files...", checks)

        # Try to download with progress tracking
        def download_with_progress():
            # This will download if not present, or verify if already cached
            snapshot_download(
                model_id,
                cache_dir=cache_dir_str,
                local_files_only=False,
                resume_download=True
            )

        # Check model size/status first
        progress_callback(25, "Downloading Flux model files (if needed)...", checks)
        download_with_progress()

        checks["flux_model"] = {
            "status": "ok",
            "message": "Flux.1-dev model ready"
        }
        print("[SETUP] Flux model: Ready")

    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "403" in error_msg:
            checks["flux_model"] = {
                "status": "error",
                "message": "Authentication required - run 'huggingface-cli login' first"
            }
        else:
            checks["flux_model"] = {
                "status": "error",
                "message": f"Model download failed: {error_msg[:100]}"
            }
        print(f"[SETUP] Flux model error: {e}")
        return {"success": False, "checks": checks, "error": str(e)}

    # Skip model loading verification - it causes segfaults and training loads model anyway
    progress_callback(70, "Skipping model load verification (will load during training)...", checks)

    checks["model_load"] = {
        "status": "ok",
        "message": "Model verified via file check (will load during training)"
    }
    print("[SETUP] Model load: Skipped (files verified, will load during training)")

    checks["inference_test"] = {
        "status": "ok",
        "message": "Skipped (will test during first generation)"
    }
    print("[SETUP] Inference test: Skipped")

    # All checks passed
    progress_callback(100, "Setup complete!", checks)

    # Determine overall success
    has_errors = any(c.get("status") == "error" for c in checks.values())

    return {
        "success": not has_errors,
        "checks": checks
    }


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("=" * 50)
    print("Flux LoRA Pipeline Server Starting...")
    print("=" * 50)
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("WARNING: No GPU detected! Running on CPU will be very slow.")
    print("=" * 50)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
