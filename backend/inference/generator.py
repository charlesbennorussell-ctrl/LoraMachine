import os

# Disable xformers to prevent segfaults
os.environ["XFORMERS_DISABLED"] = "1"
os.environ["DIFFUSERS_NO_XFORMERS"] = "1"

import torch
from pathlib import Path
from PIL import Image
import uuid
from typing import Optional
import asyncio
import gc


class FluxGenerator:
    """
    Flux image generator with LoRA support.
    Optimized for RTX 4080 (16GB VRAM).
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = None
        self.current_lora_path = None
        self.current_lora_strength = None

    def _load_pipeline(self):
        """Load the Flux pipeline with memory optimizations."""
        if self.pipeline is None:
            from diffusers import FluxPipeline

            print("Loading Flux pipeline...")
            print("This may take a few minutes on first run...")

            # Use D: drive for cache to avoid C: drive space issues
            cache_dir = "D:/CTRL_ITERATION/flux-cache"
            model_id = "black-forest-labs/FLUX.1-dev"

            # WORKAROUND: Load components individually to avoid Windows segfault
            # FluxPipeline.from_pretrained() crashes on Windows when loading all components together
            print("Loading Flux components individually (Windows workaround)...")

            from diffusers import FlowMatchEulerDiscreteScheduler, AutoencoderKL
            from diffusers.models import FluxTransformer2DModel
            from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

            print("  Loading scheduler...")
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                model_id, subfolder="scheduler", cache_dir=cache_dir
            )

            print("  Loading VAE...")
            vae = AutoencoderKL.from_pretrained(
                model_id, subfolder="vae", torch_dtype=torch.bfloat16, cache_dir=cache_dir
            )

            print("  Loading CLIP text encoder...")
            text_encoder = CLIPTextModel.from_pretrained(
                model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16, cache_dir=cache_dir
            )
            tokenizer = CLIPTokenizer.from_pretrained(
                model_id, subfolder="tokenizer", cache_dir=cache_dir
            )

            print("  Loading T5 text encoder...")
            text_encoder_2 = T5EncoderModel.from_pretrained(
                model_id, subfolder="text_encoder_2", torch_dtype=torch.bfloat16, cache_dir=cache_dir
            )
            tokenizer_2 = T5TokenizerFast.from_pretrained(
                model_id, subfolder="tokenizer_2", cache_dir=cache_dir
            )

            print("  Loading transformer...")
            transformer = FluxTransformer2DModel.from_pretrained(
                model_id, subfolder="transformer", torch_dtype=torch.bfloat16, cache_dir=cache_dir
            )

            print("  Assembling pipeline...")
            self.pipeline = FluxPipeline(
                scheduler=scheduler,
                vae=vae,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                transformer=transformer
            )

            # Memory optimizations for 16GB VRAM
            self.pipeline.enable_model_cpu_offload()
            self.pipeline.enable_vae_slicing()
            self.pipeline.enable_vae_tiling()

            # NOTE: xformers disabled to prevent segfaults with incompatible triton
            # Using PyTorch's native scaled_dot_product_attention instead
            print("Using PyTorch native attention (xformers disabled for stability)")

            print("Flux pipeline loaded successfully!")

    def _load_lora(self, lora_path: str, strength: float):
        """Load LoRA weights with specified strength."""
        lora_path = Path(lora_path)

        if not lora_path.exists():
            raise ValueError(f"LoRA path does not exist: {lora_path}")

        # Check if we need to reload
        if self.current_lora_path != str(lora_path):
            print(f"Loading LoRA from {lora_path}...")

            # Unload previous LoRA if any
            if self.current_lora_path is not None:
                try:
                    self.pipeline.unload_lora_weights()
                except Exception:
                    pass

            # Load new LoRA
            try:
                self.pipeline.load_lora_weights(
                    str(lora_path),
                    adapter_name="default"
                )
                self.current_lora_path = str(lora_path)
                print(f"LoRA loaded: {lora_path.name}")
            except Exception as e:
                print(f"Error loading LoRA: {e}")
                # Try alternative loading method
                try:
                    from peft import PeftModel
                    # Load as PEFT adapter
                    self.pipeline.transformer = PeftModel.from_pretrained(
                        self.pipeline.transformer,
                        str(lora_path)
                    )
                    self.current_lora_path = str(lora_path)
                    print(f"LoRA loaded via PEFT: {lora_path.name}")
                except Exception as e2:
                    raise ValueError(f"Could not load LoRA: {e}, {e2}")

        # Set LoRA scale (strength)
        if self.current_lora_strength != strength:
            try:
                self.pipeline.set_adapters(["default"], adapter_weights=[strength])
            except Exception:
                # Alternative method
                try:
                    for name, module in self.pipeline.transformer.named_modules():
                        if hasattr(module, "scale"):
                            module.scale = strength
                except Exception:
                    pass
            self.current_lora_strength = strength
            print(f"LoRA strength set to {strength}")

    def _unload_lora(self):
        """Unload current LoRA weights."""
        if self.current_lora_path is not None:
            try:
                self.pipeline.unload_lora_weights()
            except Exception:
                pass
            self.current_lora_path = None
            self.current_lora_strength = None

    async def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        lora_path: Optional[str] = None,
        lora_strength: float = 0.8,
        steps: int = 28,
        guidance_scale: float = 3.5,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        output_dir: str = "./outputs/generated"
    ) -> str:
        """
        Generate an image with optional LoRA.

        Args:
            prompt: Text prompt for generation
            negative_prompt: Negative prompt (things to avoid)
            lora_path: Path to LoRA weights directory
            lora_strength: LoRA influence strength (0.0-1.0)
            steps: Number of inference steps
            guidance_scale: Classifier-free guidance scale
            width: Output image width
            height: Output image height
            seed: Random seed for reproducibility
            output_dir: Directory to save generated images

        Returns:
            Path to the generated image
        """
        # Load pipeline if needed
        self._load_pipeline()

        # Load LoRA if specified
        if lora_path:
            self._load_lora(lora_path, lora_strength)
        else:
            self._unload_lora()

        # Set seed
        if seed is None:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()

        generator = torch.Generator(device="cpu").manual_seed(seed)

        print(f"Generating image...")
        print(f"  Prompt: {prompt[:80]}...")
        print(f"  Seed: {seed}")
        print(f"  Steps: {steps}")
        print(f"  Size: {width}x{height}")
        if lora_path:
            print(f"  LoRA: {Path(lora_path).name} @ {lora_strength}")

        # Generate
        with torch.inference_mode():
            result = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator,
            )
            image = result.images[0]

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate filename with metadata
        lora_name = Path(lora_path).name if lora_path else "no_lora"
        filename = f"{uuid.uuid4().hex[:8]}_s{lora_strength:.1f}_{lora_name}_seed{seed}.png"
        save_path = output_path / filename

        # Save image
        image.save(save_path, quality=95)
        print(f"Image saved: {save_path}")

        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return str(save_path)

    async def generate_batch(
        self,
        prompts: list,
        **kwargs
    ) -> list:
        """Generate multiple images with the same settings."""
        results = []
        for i, prompt in enumerate(prompts):
            print(f"Generating image {i+1}/{len(prompts)}...")
            path = await self.generate(prompt=prompt, **kwargs)
            results.append(path)
            await asyncio.sleep(0.1)  # Small delay between generations
        return results

    def unload(self):
        """Unload the pipeline to free memory."""
        if self.pipeline is not None:
            self._unload_lora()
            del self.pipeline
            self.pipeline = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Pipeline unloaded")


class FluxGeneratorDev:
    """
    Development/mock generator for testing without GPU.
    Generates placeholder images.
    """

    def __init__(self):
        self.device = "cpu"

    async def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        lora_path: Optional[str] = None,
        lora_strength: float = 0.8,
        steps: int = 28,
        guidance_scale: float = 3.5,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        output_dir: str = "./outputs/generated"
    ) -> str:
        """Generate a placeholder image for testing."""
        from PIL import Image, ImageDraw, ImageFont
        import random

        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        random.seed(seed)

        # Create a gradient background
        img = Image.new("RGB", (width, height))
        pixels = img.load()

        # Random gradient colors
        r1, g1, b1 = random.randint(50, 150), random.randint(50, 150), random.randint(100, 200)
        r2, g2, b2 = random.randint(100, 200), random.randint(50, 150), random.randint(50, 150)

        for y in range(height):
            for x in range(width):
                ratio_x = x / width
                ratio_y = y / height
                ratio = (ratio_x + ratio_y) / 2

                r = int(r1 * (1 - ratio) + r2 * ratio)
                g = int(g1 * (1 - ratio) + g2 * ratio)
                b = int(b1 * (1 - ratio) + b2 * ratio)
                pixels[x, y] = (r, g, b)

        # Add text
        draw = ImageDraw.Draw(img)
        text = f"DEV MODE\nPrompt: {prompt[:50]}...\nLoRA: {lora_strength:.1f}\nSeed: {seed}"

        # Use default font
        draw.multiline_text(
            (width // 2, height // 2),
            text,
            fill=(255, 255, 255),
            anchor="mm",
            align="center"
        )

        # Save
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        lora_name = Path(lora_path).name if lora_path else "no_lora"
        filename = f"{uuid.uuid4().hex[:8]}_s{lora_strength:.1f}_{lora_name}_seed{seed}.png"
        save_path = output_path / filename

        img.save(save_path)
        print(f"[DEV] Placeholder image saved: {save_path}")

        await asyncio.sleep(0.5)  # Simulate generation time
        return str(save_path)
