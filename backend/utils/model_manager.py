"""
Model management utilities for the Flux LoRA Pipeline.
Handles model downloads, caching, and verification.
"""

from pathlib import Path
from typing import Optional, Dict, List
import json
import hashlib


class ModelManager:
    """
    Manages ML models for the pipeline.
    Handles downloads, caching, and version tracking.
    """

    def __init__(self, models_dir: str = "./models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Sub-directories
        self.flux_dir = self.models_dir / "flux"
        self.loras_dir = self.models_dir / "loras"
        self.refiners_dir = self.models_dir / "refiners"

        for d in [self.flux_dir, self.loras_dir, self.refiners_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Model registry
        self.registry_path = self.models_dir / "registry.json"
        self.registry = self._load_registry()

    def _load_registry(self) -> Dict:
        """Load model registry from disk."""
        if self.registry_path.exists():
            with open(self.registry_path, "r") as f:
                return json.load(f)
        return {"models": {}, "loras": {}}

    def _save_registry(self):
        """Save model registry to disk."""
        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=2)

    def register_lora(
        self,
        name: str,
        path: str,
        trigger_word: str,
        base_model: str = "FLUX.1-dev",
        metadata: Optional[Dict] = None
    ):
        """
        Register a trained LoRA in the registry.

        Args:
            name: LoRA name
            path: Path to LoRA weights
            trigger_word: Trigger word for activation
            base_model: Base model the LoRA was trained on
            metadata: Additional metadata
        """
        self.registry["loras"][name] = {
            "path": str(path),
            "trigger_word": trigger_word,
            "base_model": base_model,
            "metadata": metadata or {}
        }
        self._save_registry()

    def get_lora(self, name: str) -> Optional[Dict]:
        """Get LoRA info by name."""
        return self.registry["loras"].get(name)

    def list_loras(self) -> List[Dict]:
        """List all registered LoRAs."""
        loras = []
        for name, info in self.registry["loras"].items():
            loras.append({
                "name": name,
                **info
            })
        return loras

    def delete_lora(self, name: str) -> bool:
        """Delete a LoRA from registry and disk."""
        if name not in self.registry["loras"]:
            return False

        lora_info = self.registry["loras"][name]
        lora_path = Path(lora_info["path"])

        # Remove from disk
        if lora_path.exists():
            import shutil
            if lora_path.is_dir():
                shutil.rmtree(lora_path)
            else:
                lora_path.unlink()

        # Remove from registry
        del self.registry["loras"][name]
        self._save_registry()

        return True

    def get_flux_status(self) -> Dict:
        """Check if Flux model is downloaded and ready."""
        # Check for model files
        has_model = False
        model_size = 0

        for f in self.flux_dir.rglob("*.safetensors"):
            has_model = True
            model_size += f.stat().st_size

        for f in self.flux_dir.rglob("*.bin"):
            has_model = True
            model_size += f.stat().st_size

        return {
            "downloaded": has_model,
            "path": str(self.flux_dir),
            "size_gb": model_size / (1024**3) if model_size > 0 else 0
        }

    def get_refiner_status(self) -> Dict:
        """Check refinement model status."""
        realesrgan_path = self.refiners_dir / "RealESRGAN_x4plus.pth"
        gfpgan_path = self.refiners_dir / "GFPGANv1.4.pth"

        return {
            "realesrgan": {
                "downloaded": realesrgan_path.exists(),
                "path": str(realesrgan_path)
            },
            "gfpgan": {
                "downloaded": gfpgan_path.exists(),
                "path": str(gfpgan_path)
            }
        }

    async def download_flux(self, progress_callback=None):
        """
        Download Flux model from HuggingFace.
        Requires user to be logged in via huggingface-cli.
        """
        from huggingface_hub import snapshot_download

        if progress_callback:
            await progress_callback(0, "Starting Flux download...")

        try:
            snapshot_download(
                "black-forest-labs/FLUX.1-dev",
                local_dir=str(self.flux_dir),
                local_dir_use_symlinks=False,
                resume_download=True
            )

            if progress_callback:
                await progress_callback(100, "Flux download complete!")

            return True
        except Exception as e:
            if progress_callback:
                await progress_callback(0, f"Error: {str(e)}")
            return False

    async def download_realesrgan(self, progress_callback=None):
        """Download Real-ESRGAN model."""
        import urllib.request

        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        output_path = self.refiners_dir / "RealESRGAN_x4plus.pth"

        if output_path.exists():
            if progress_callback:
                await progress_callback(100, "Already downloaded")
            return True

        try:
            if progress_callback:
                await progress_callback(0, "Downloading Real-ESRGAN...")

            urllib.request.urlretrieve(url, str(output_path))

            if progress_callback:
                await progress_callback(100, "Download complete!")

            return True
        except Exception as e:
            if progress_callback:
                await progress_callback(0, f"Error: {str(e)}")
            return False

    def verify_lora(self, path: str) -> Dict:
        """
        Verify a LoRA is valid and get its info.

        Args:
            path: Path to LoRA directory

        Returns:
            Dictionary with verification results
        """
        lora_path = Path(path)

        if not lora_path.exists():
            return {"valid": False, "error": "Path does not exist"}

        # Check for required files
        has_config = (lora_path / "adapter_config.json").exists() or \
                    (lora_path / "config.json").exists()

        has_weights = any(lora_path.glob("*.safetensors")) or \
                     any(lora_path.glob("*.bin"))

        if not has_config:
            return {"valid": False, "error": "No config file found"}

        if not has_weights:
            return {"valid": False, "error": "No weight files found"}

        # Try to load config
        config = {}
        for config_name in ["adapter_config.json", "config.json", "training_config.json"]:
            config_path = lora_path / config_name
            if config_path.exists():
                with open(config_path, "r") as f:
                    config = json.load(f)
                break

        return {
            "valid": True,
            "path": str(lora_path),
            "config": config
        }


def get_gpu_info() -> Dict:
    """Get GPU information."""
    import torch

    if not torch.cuda.is_available():
        return {
            "available": False,
            "name": None,
            "memory_total": 0,
            "memory_free": 0
        }

    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "memory_total": torch.cuda.get_device_properties(0).total_memory / (1024**3),
        "memory_free": torch.cuda.mem_get_info()[0] / (1024**3),
        "cuda_version": torch.version.cuda
    }


def estimate_training_time(
    num_images: int,
    steps: int,
    resolution: int,
    gpu_name: str = "RTX 4080"
) -> Dict:
    """
    Estimate training time based on parameters.

    Args:
        num_images: Number of training images
        steps: Number of training steps
        resolution: Training resolution
        gpu_name: GPU name for estimation

    Returns:
        Dictionary with time estimates
    """
    # Base time per step (seconds) for different GPUs at 1024 resolution
    gpu_speeds = {
        "RTX 4080": 0.8,
        "RTX 4090": 0.5,
        "RTX 3090": 1.0,
        "RTX 3080": 1.2,
        "A100": 0.3,
    }

    base_time = gpu_speeds.get(gpu_name, 1.0)

    # Adjust for resolution
    resolution_factor = (resolution / 1024) ** 2

    # Calculate
    time_per_step = base_time * resolution_factor
    total_seconds = steps * time_per_step

    return {
        "estimated_seconds": total_seconds,
        "estimated_minutes": total_seconds / 60,
        "estimated_hours": total_seconds / 3600,
        "time_per_step": time_per_step
    }
