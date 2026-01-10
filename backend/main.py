import torch
import os
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import json
from pathlib import Path
import shutil

# Set cache directories to D: drive to avoid C: drive space issues
os.environ['HF_HOME'] = 'D:/CTRL_ITERATION/flux-cache'
os.environ['TRANSFORMERS_CACHE'] = 'D:/CTRL_ITERATION/flux-cache'
os.environ['HF_DATASETS_CACHE'] = 'D:/CTRL_ITERATION/flux-cache'

from training.trainer import LoRATrainer
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
async def start_training(config: TrainingConfig, background_tasks: BackgroundTasks):
    """Start LoRA training job"""
    global training_status

    async def train_task():
        global training_status
        training_status = {"active": True, "progress": 0, "message": "Initializing..."}
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
                progress_callback=lambda p, m: asyncio.create_task(
                    broadcast({"type": "training_status", "data": {"active": True, "progress": p, "message": m}})
                )
            )
            training_status = {"active": False, "progress": 100, "message": "Complete!"}
        except Exception as e:
            training_status = {"active": False, "progress": 0, "message": f"Error: {str(e)}"}
            import traceback
            traceback.print_exc()

        await broadcast({"type": "training_status", "data": training_status})

    background_tasks.add_task(lambda: asyncio.run(train_task()))
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
async def run_iteration(config: IterationConfig, background_tasks: BackgroundTasks):
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

    background_tasks.add_task(lambda: asyncio.run(iterate_task()))
    return {"status": "Iteration started"}


@app.post("/like/{image_id}")
async def like_image(image_id: str, background_tasks: BackgroundTasks):
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

        background_tasks.add_task(lambda: asyncio.run(refine_task()))
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
        "gpu_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_memory": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB" if torch.cuda.is_available() else None
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
