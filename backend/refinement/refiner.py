import torch
from pathlib import Path
from PIL import Image
import uuid
from typing import Optional, List, Dict
import asyncio


class ImageRefiner:
    """
    Image refinement using Real-ESRGAN or similar upscaling/enhancement.
    Provides lightweight but effective image enhancement for liked images.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.model_type = None

    def _load_model(self):
        """Load the upscaling model."""
        if self.model is not None:
            return

        # Try Real-ESRGAN first
        try:
            from realesrgan import RealESRGANer
            from basicsr.archs.rrdbnet_arch import RRDBNet

            print("Loading Real-ESRGAN model...")

            # Model architecture for RealESRGAN_x4plus
            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4
            )

            model_path = Path(__file__).parent.parent.parent / "models" / "refiners" / "RealESRGAN_x4plus.pth"

            if not model_path.exists():
                print(f"Model not found at {model_path}")
                print("Attempting to download...")
                self._download_model(model_path)

            self.model = RealESRGANer(
                scale=4,
                model_path=str(model_path),
                dni_weight=None,
                model=model,
                tile=400,  # Tile size for memory efficiency
                tile_pad=10,
                pre_pad=0,
                half=True if self.device == "cuda" else False,
                device=self.device
            )
            self.model_type = "realesrgan"
            print("Real-ESRGAN loaded successfully!")

        except ImportError as e:
            print(f"Real-ESRGAN not available: {e}")
            print("Using Pillow-based upscaling fallback")
            self.model = "pillow_fallback"
            self.model_type = "pillow"

        except Exception as e:
            print(f"Error loading Real-ESRGAN: {e}")
            print("Using Pillow-based upscaling fallback")
            self.model = "pillow_fallback"
            self.model_type = "pillow"

    def _download_model(self, model_path: Path):
        """Download the Real-ESRGAN model."""
        import urllib.request

        model_path.parent.mkdir(parents=True, exist_ok=True)

        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

        print(f"Downloading Real-ESRGAN model from {url}...")
        try:
            urllib.request.urlretrieve(url, str(model_path))
            print("Download complete!")
        except Exception as e:
            print(f"Download failed: {e}")
            raise

    async def refine(
        self,
        image_path: str,
        output_dir: str = "./outputs/refined",
        scale: float = 2.0,
        denoise_strength: float = 0.5
    ) -> str:
        """
        Refine/upscale an image.

        Args:
            image_path: Path to input image
            output_dir: Directory to save refined image
            scale: Upscale factor (will be adjusted based on model)
            denoise_strength: Denoising strength (0-1)

        Returns:
            Path to refined image
        """
        self._load_model()

        input_path = Path(image_path)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Refining image: {input_path.name}")

        # Load image
        img = Image.open(input_path).convert("RGB")
        original_size = img.size

        if self.model_type == "pillow":
            # Fallback: High-quality Lanczos upscaling
            new_width = int(img.width * scale)
            new_height = int(img.height * scale)
            refined = img.resize((new_width, new_height), Image.LANCZOS)

            # Apply simple sharpening
            from PIL import ImageFilter, ImageEnhance
            refined = refined.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

            # Slight contrast enhancement
            enhancer = ImageEnhance.Contrast(refined)
            refined = enhancer.enhance(1.05)

        else:
            # Real-ESRGAN upscaling
            import numpy as np
            import cv2

            # Convert to numpy for Real-ESRGAN
            img_np = np.array(img)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # Run enhancement
            output, _ = self.model.enhance(img_np, outscale=scale)

            # Convert back to PIL
            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            refined = Image.fromarray(output)

        # Save refined image
        filename = f"refined_{input_path.stem}_{uuid.uuid4().hex[:6]}.png"
        save_path = output_path / filename
        refined.save(save_path, quality=95)

        print(f"Refined image saved: {save_path}")
        print(f"  Original size: {original_size[0]}x{original_size[1]}")
        print(f"  Refined size: {refined.size[0]}x{refined.size[1]}")

        return str(save_path)

    async def batch_refine(
        self,
        image_paths: List[str],
        output_dir: str = "./outputs/refined",
        progress_callback=None
    ) -> List[Dict]:
        """
        Refine multiple images.

        Args:
            image_paths: List of image paths to refine
            output_dir: Output directory
            progress_callback: Progress callback function

        Returns:
            List of dictionaries with original and refined paths
        """
        results = []

        for i, path in enumerate(image_paths):
            try:
                refined_path = await self.refine(path, output_dir)
                results.append({
                    "original": path,
                    "refined": refined_path,
                    "success": True
                })
            except Exception as e:
                print(f"Error refining {path}: {e}")
                results.append({
                    "original": path,
                    "refined": None,
                    "success": False,
                    "error": str(e)
                })

            if progress_callback:
                await progress_callback(i + 1, len(image_paths), results[-1])

            await asyncio.sleep(0.1)

        return results

    def unload(self):
        """Unload the model to free memory."""
        if self.model is not None and self.model != "pillow_fallback":
            del self.model
            self.model = None
            self.model_type = None
            torch.cuda.empty_cache()
            print("Refiner model unloaded")


class FaceEnhancer:
    """
    Face-specific enhancement using GFPGAN.
    Use this for portrait images with faces.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None

    def _load_model(self):
        """Load GFPGAN model."""
        if self.model is not None:
            return

        try:
            from gfpgan import GFPGANer

            model_path = Path(__file__).parent.parent.parent / "models" / "refiners" / "GFPGANv1.4.pth"

            if not model_path.exists():
                print("GFPGAN model not found. Face enhancement not available.")
                self.model = None
                return

            self.model = GFPGANer(
                model_path=str(model_path),
                upscale=2,
                arch='clean',
                channel_multiplier=2,
                device=self.device
            )
            print("GFPGAN loaded successfully!")

        except ImportError:
            print("GFPGAN not available")
            self.model = None

    async def enhance_face(
        self,
        image_path: str,
        output_dir: str = "./outputs/refined"
    ) -> Optional[str]:
        """
        Enhance faces in an image.

        Args:
            image_path: Path to input image
            output_dir: Output directory

        Returns:
            Path to enhanced image, or None if enhancement failed
        """
        self._load_model()

        if self.model is None:
            print("GFPGAN not available, skipping face enhancement")
            return None

        import cv2
        import numpy as np

        input_path = Path(image_path)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Load image
        img = cv2.imread(str(input_path))

        # Enhance
        _, _, output = self.model.enhance(
            img,
            has_aligned=False,
            only_center_face=False,
            paste_back=True
        )

        # Save
        filename = f"face_enhanced_{input_path.stem}_{uuid.uuid4().hex[:6]}.png"
        save_path = output_path / filename
        cv2.imwrite(str(save_path), output)

        return str(save_path)


class CompositeRefiner:
    """
    Combines multiple refinement techniques for best results.
    """

    def __init__(self):
        self.image_refiner = ImageRefiner()
        self.face_enhancer = FaceEnhancer()

    async def refine(
        self,
        image_path: str,
        output_dir: str = "./outputs/refined",
        enhance_faces: bool = True,
        upscale: bool = True,
        scale: float = 2.0
    ) -> str:
        """
        Apply composite refinement.

        Args:
            image_path: Path to input image
            output_dir: Output directory
            enhance_faces: Whether to enhance faces
            upscale: Whether to upscale
            scale: Upscale factor

        Returns:
            Path to refined image
        """
        current_path = image_path

        # Face enhancement first (works better on original resolution)
        if enhance_faces:
            face_path = await self.face_enhancer.enhance_face(current_path, output_dir)
            if face_path:
                current_path = face_path

        # Then upscale
        if upscale:
            refined_path = await self.image_refiner.refine(
                current_path,
                output_dir,
                scale=scale
            )
            current_path = refined_path

        return current_path
