"""
GGUF-based Flux Img2Img Generator - Optimized for 16GB VRAM

Uses Q8_0 GGUF quantized transformer (~12GB) instead of full FP16 (~24GB).
This allows running on 16GB VRAM cards like RTX 4080 with room to spare.

Based on: https://huggingface.co/docs/diffusers/quantization/gguf
"""

import os

# Disable xformers to prevent segfaults on Windows
os.environ["XFORMERS_DISABLED"] = "1"
os.environ["DIFFUSERS_NO_XFORMERS"] = "1"

# DISABLE GGUF CUDA kernels - no Windows build available
# See: https://github.com/huggingface/diffusers/issues/10795
os.environ["DIFFUSERS_GGUF_CUDA_KERNELS"] = "false"

import torch
from pathlib import Path
from PIL import Image
import uuid
from typing import Optional, Union
import asyncio
import gc


class FluxImg2ImgGenerator:
    """
    GGUF-quantized Flux image-to-image generator with LoRA support.

    OPTIMIZED FOR 16GB VRAM:
    - Uses Q8_0 GGUF quantized transformer (~12GB vs 24GB full)
    - CPU offloading for other components
    - VAE slicing and tiling enabled

    Key parameters:
    - lora_strength: How much the LoRA influences the style (0-1)
    - creativity: Denoising strength - how much to change from input (0-1)
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = None
        self.current_lora_path = None
        self.current_lora_strength = None

        # Paths
        self.cache_dir = "D:/CTRL_ITERATION/flux-cache"
        self.gguf_model_path = "D:/COMFY UI/StabilityMatrix-win-x64/Data/Packages/ComfyUI/models/unet/flux1-dev-Q8_0.gguf"
        self.model_id = "black-forest-labs/FLUX.1-dev"

    def _load_pipeline(self):
        """Load the GGUF-quantized Flux img2img pipeline."""
        if self.pipeline is not None:
            return

        from diffusers import (
            FluxImg2ImgPipeline,
            FluxTransformer2DModel,
            GGUFQuantizationConfig,
            AutoencoderKL,
            FlowMatchEulerDiscreteScheduler
        )
        from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

        print("=" * 60)
        print("Loading GGUF-Quantized Flux Img2Img Pipeline")
        print("=" * 60)
        print(f"GGUF Model: {self.gguf_model_path}")
        print("This uses ~12GB VRAM instead of ~24GB!")
        print("=" * 60)

        # Verify GGUF file exists
        if not Path(self.gguf_model_path).exists():
            raise FileNotFoundError(
                f"GGUF model not found at: {self.gguf_model_path}\n"
                "Please ensure you have flux1-dev-Q8_0.gguf in your ComfyUI models folder."
            )

        try:
            # Step 1: Load GGUF quantized transformer (the big one - but quantized!)
            print("\n[1/7] Loading GGUF quantized transformer...")
            print("      (This is the magic - Q8 = ~12GB instead of 24GB)")

            transformer = FluxTransformer2DModel.from_single_file(
                self.gguf_model_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
                torch_dtype=torch.bfloat16,
            )
            print(f"      Transformer loaded! VRAM: {torch.cuda.memory_allocated() / 1e9:.2f}GB")

            # Step 2: Load scheduler
            print("\n[2/7] Loading scheduler...")
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                self.model_id,
                subfolder="scheduler",
                cache_dir=self.cache_dir
            )

            # Step 3: Load VAE
            print("\n[3/7] Loading VAE...")
            vae = AutoencoderKL.from_pretrained(
                self.model_id,
                subfolder="vae",
                torch_dtype=torch.bfloat16,
                cache_dir=self.cache_dir
            )

            # Step 4: Load CLIP text encoder
            print("\n[4/7] Loading CLIP text encoder...")
            text_encoder = CLIPTextModel.from_pretrained(
                self.model_id,
                subfolder="text_encoder",
                torch_dtype=torch.bfloat16,
                cache_dir=self.cache_dir
            )

            # Step 5: Load CLIP tokenizer
            print("\n[5/7] Loading CLIP tokenizer...")
            tokenizer = CLIPTokenizer.from_pretrained(
                self.model_id,
                subfolder="tokenizer",
                cache_dir=self.cache_dir
            )

            # Step 6: Load T5 text encoder
            print("\n[6/7] Loading T5 text encoder...")
            text_encoder_2 = T5EncoderModel.from_pretrained(
                self.model_id,
                subfolder="text_encoder_2",
                torch_dtype=torch.bfloat16,
                cache_dir=self.cache_dir
            )

            # Step 7: Load T5 tokenizer
            print("\n[7/7] Loading T5 tokenizer...")
            tokenizer_2 = T5TokenizerFast.from_pretrained(
                self.model_id,
                subfolder="tokenizer_2",
                cache_dir=self.cache_dir
            )

            # Assemble pipeline with GGUF transformer
            print("\nAssembling Img2Img pipeline with GGUF transformer...")
            self.pipeline = FluxImg2ImgPipeline(
                scheduler=scheduler,
                vae=vae,
                text_encoder=text_encoder,
                text_encoder_2=text_encoder_2,
                tokenizer=tokenizer,
                tokenizer_2=tokenizer_2,
                transformer=transformer
            )

            # Memory optimizations
            print("Enabling memory optimizations...")
            self.pipeline.enable_model_cpu_offload()

            # VAE optimizations
            if hasattr(self.pipeline.vae, 'enable_slicing'):
                self.pipeline.vae.enable_slicing()
            if hasattr(self.pipeline.vae, 'enable_tiling'):
                self.pipeline.vae.enable_tiling()

            vram_used = torch.cuda.memory_allocated() / 1e9
            print("\n" + "=" * 60)
            print(f"Pipeline loaded successfully!")
            print(f"VRAM after loading: {vram_used:.2f}GB")
            print("=" * 60 + "\n")

        except Exception as e:
            print(f"\nERROR loading pipeline: {e}")
            import traceback
            traceback.print_exc()
            raise

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

            # Try multiple loading methods
            loaded = False
            errors = []

            # Method 1: Load directory directly
            try:
                self.pipeline.load_lora_weights(str(lora_path), adapter_name="default")
                self.current_lora_path = str(lora_path)
                loaded = True
                print(f"LoRA loaded (directory): {lora_path.name}")
            except Exception as e:
                errors.append(f"Directory load: {e}")

            # Method 2: Load safetensors file directly
            if not loaded:
                safetensors_file = lora_path / "pytorch_lora_weights.safetensors"
                if safetensors_file.exists():
                    try:
                        self.pipeline.load_lora_weights(str(safetensors_file), adapter_name="default")
                        self.current_lora_path = str(lora_path)
                        loaded = True
                        print(f"LoRA loaded (safetensors): {lora_path.name}")
                    except Exception as e:
                        errors.append(f"Safetensors load: {e}")

            if not loaded:
                print(f"Warning: Could not load LoRA. Errors: {errors}")
                print("Continuing without LoRA...")
                return

        # Set LoRA scale (strength)
        if self.current_lora_strength != strength:
            try:
                self.pipeline.set_adapters(["default"], adapter_weights=[strength])
                self.current_lora_strength = strength
                print(f"LoRA strength set to {strength}")
            except Exception as e:
                print(f"Warning: Could not set LoRA strength: {e}")

    def _unload_lora(self):
        """Unload current LoRA weights."""
        if self.current_lora_path is not None:
            try:
                self.pipeline.unload_lora_weights()
            except Exception:
                pass
            self.current_lora_path = None
            self.current_lora_strength = None

    def _prepare_image(self, image: Union[str, Path, Image.Image], target_size: tuple) -> Image.Image:
        """Load and prepare input image for img2img."""
        if isinstance(image, (str, Path)):
            image = Image.open(image)

        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize to target dimensions
        image = image.resize(target_size, Image.Resampling.LANCZOS)

        return image

    async def generate(
        self,
        input_image: Union[str, Path, Image.Image],
        prompt: str,
        negative_prompt: str = "",
        lora_path: Optional[str] = None,
        lora_strength: float = 1.0,
        creativity: float = 0.5,
        steps: int = 28,
        guidance_scale: float = 3.5,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        output_dir: str = "./outputs/generated"
    ) -> str:
        """
        Generate an image using img2img with optional LoRA.

        Args:
            input_image: Input image (path or PIL Image)
            prompt: Text prompt for generation
            negative_prompt: Negative prompt (things to avoid)
            lora_path: Path to LoRA weights directory
            lora_strength: LoRA influence strength (0.0-1.0)
            creativity: Denoising strength (0.0-1.0) - higher = more change
            steps: Number of inference steps
            guidance_scale: Classifier-free guidance scale
            width: Output image width
            height: Output image height
            seed: Random seed for reproducibility
            output_dir: Directory to save generated images

        Returns:
            Path to the generated image
        """
        # Load pipeline
        self._load_pipeline()

        # Load LoRA if specified
        if lora_path:
            self._load_lora(lora_path, lora_strength)
        else:
            self._unload_lora()

        # Prepare input image
        input_img = self._prepare_image(input_image, (width, height))

        # Set seed
        if seed is None:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()

        generator = torch.Generator(device="cpu").manual_seed(seed)

        print(f"\n{'='*60}")
        print(f"Generating img2img (GGUF Q8)")
        print(f"  Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        print(f"  Creativity: {creativity}")
        print(f"  Seed: {seed}")
        print(f"  Steps: {steps}")
        print(f"  Size: {width}x{height}")
        if lora_path:
            print(f"  LoRA: {Path(lora_path).name} @ {lora_strength}")
        print(f"{'='*60}")

        # Validate creativity - must be > 0 to have at least 1 step
        if creativity < 0.05:
            creativity = 0.05
            print(f"  (creativity adjusted to minimum 0.05)")

        # Generate
        try:
            with torch.inference_mode():
                result = self.pipeline(
                    prompt=prompt,
                    image=input_img,
                    strength=creativity,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )
                image = result.images[0]

        except torch.cuda.OutOfMemoryError:
            print("OOM! Clearing cache and retrying at lower resolution...")
            gc.collect()
            torch.cuda.empty_cache()

            # Retry with smaller size
            smaller_size = (width // 2, height // 2)
            input_img_small = self._prepare_image(input_image, smaller_size)

            with torch.inference_mode():
                result = self.pipeline(
                    prompt=prompt,
                    image=input_img_small,
                    strength=creativity,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                )
                image = result.images[0].resize((width, height), Image.Resampling.LANCZOS)

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate filename
        lora_name = Path(lora_path).name if lora_path else "no_lora"
        filename = f"{uuid.uuid4().hex[:8]}_c{creativity:.2f}_s{lora_strength:.1f}_{lora_name}_seed{seed}.png"
        save_path = output_path / filename

        # Save image
        image.save(save_path, quality=95)
        print(f"Image saved: {save_path}")

        # Cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return str(save_path)

    async def generate_creativity_sweep(
        self,
        input_image: Union[str, Path, Image.Image],
        prompt: str,
        lora_path: str,
        creativity_values: list = None,
        lora_strength: float = 1.0,
        seed: Optional[int] = None,
        output_dir: str = "./outputs/generated",
        progress_callback=None,
        **kwargs
    ) -> list:
        """
        Generate multiple images at different creativity levels.
        """
        if creativity_values is None:
            # Note: creativity=0.0 doesn't work (results in 0 steps)
            # Start at 0.1 minimum for valid generation
            creativity_values = [0.1, 0.3, 0.5, 0.7, 0.9]

        if seed is None:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()

        results = []

        print(f"\n{'='*60}")
        print(f"Creativity Sweep (GGUF Q8 img2img)")
        print(f"{'='*60}")
        print(f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        print(f"LoRA: {lora_path} @ {lora_strength}")
        print(f"Creativity values: {creativity_values}")
        print(f"Seed: {seed} (fixed)")
        print(f"{'='*60}\n")

        for i, creativity in enumerate(creativity_values):
            creativity = round(float(creativity), 2)
            print(f"\n[{i+1}/{len(creativity_values)}] Generating creativity={creativity}")

            try:
                image_path = await self.generate(
                    input_image=input_image,
                    prompt=prompt,
                    lora_path=lora_path,
                    lora_strength=lora_strength,
                    creativity=creativity,
                    seed=seed,
                    output_dir=output_dir,
                    **kwargs
                )

                result = {
                    "creativity": creativity,
                    "path": image_path,
                    "seed": seed,
                    "success": True
                }
                results.append(result)

                if progress_callback:
                    await progress_callback(creativity, image_path)

            except Exception as e:
                print(f"Error at creativity {creativity}: {e}")
                import traceback
                traceback.print_exc()
                results.append({
                    "creativity": creativity,
                    "path": None,
                    "seed": seed,
                    "success": False,
                    "error": str(e)
                })

            # Cleanup between generations
            gc.collect()
            torch.cuda.empty_cache()
            await asyncio.sleep(0.2)

        print(f"\n{'='*60}")
        success_count = len([r for r in results if r['success']])
        print(f"Sweep complete! {success_count}/{len(creativity_values)} images generated")
        print(f"{'='*60}\n")

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
            print("Img2img pipeline unloaded")
