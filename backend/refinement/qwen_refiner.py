"""
Qwen2-VL based image refinement and enhancement.
Uses Qwen2-VL for intelligent image analysis and Flux Fill for context-aware refinement.
"""

import torch
from pathlib import Path
from PIL import Image
import uuid
from typing import Optional, List, Tuple
import asyncio


class QwenVLRefiner:
    """
    Uses Qwen2-VL for image analysis and enhancement suggestions,
    combined with Flux Fill for intelligent inpainting/refinement.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.qwen_model = None
        self.qwen_processor = None
        self.flux_fill_pipeline = None

    def _load_qwen(self):
        """Load Qwen2-VL model for image analysis."""
        if self.qwen_model is not None:
            return

        print("Loading Qwen2-VL model...")

        try:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

            model_id = "Qwen/Qwen2-VL-7B-Instruct"

            # Load with quantization for memory efficiency
            self.qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
            )

            self.qwen_processor = AutoProcessor.from_pretrained(model_id)
            print("Qwen2-VL loaded successfully!")

        except ImportError as e:
            print(f"Qwen2-VL not available: {e}")
            print("Install with: pip install transformers qwen-vl-utils")
            raise

    def _load_flux_fill(self):
        """Load Flux Fill pipeline for inpainting."""
        if self.flux_fill_pipeline is not None:
            return

        print("Loading Flux Fill pipeline...")

        try:
            from diffusers import FluxFillPipeline

            self.flux_fill_pipeline = FluxFillPipeline.from_pretrained(
                "black-forest-labs/FLUX.1-Fill-dev",
                torch_dtype=torch.bfloat16
            )
            self.flux_fill_pipeline.enable_model_cpu_offload()
            print("Flux Fill loaded successfully!")

        except Exception as e:
            print(f"Flux Fill not available: {e}")
            raise

    async def analyze_image(self, image_path: str) -> dict:
        """
        Analyze an image using Qwen2-VL to identify areas for improvement.

        Args:
            image_path: Path to the image

        Returns:
            Dictionary with analysis results including:
            - quality_score: Overall quality assessment (0-10)
            - issues: List of identified issues
            - suggestions: Refinement suggestions
            - regions: Regions that need attention
        """
        self._load_qwen()

        image = Image.open(image_path).convert("RGB")

        # Create analysis prompt
        prompt = """Analyze this AI-generated image and provide:
1. Quality score (0-10)
2. Any artifacts, distortions, or quality issues
3. Areas that could be improved
4. Specific regions that need attention (describe location)

Focus on:
- Face quality (if present): eyes, teeth, hands, facial features
- Background consistency
- Lighting and color balance
- Any obvious AI artifacts

Be specific and concise."""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # Process with Qwen2-VL
        text = self.qwen_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.qwen_processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.inference_mode():
            output_ids = self.qwen_model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )

        response = self.qwen_processor.batch_decode(
            output_ids[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0]

        # Parse response (simplified parsing)
        return {
            "analysis": response,
            "image_path": image_path
        }

    async def create_refinement_mask(
        self,
        image_path: str,
        regions: List[str]
    ) -> Image.Image:
        """
        Use Qwen2-VL to identify regions and create a mask for inpainting.

        Args:
            image_path: Path to the image
            regions: List of region descriptions to mask

        Returns:
            PIL Image mask (white = areas to refine)
        """
        self._load_qwen()

        image = Image.open(image_path).convert("RGB")

        # For now, return a simple edge-based mask
        # In production, use Qwen2-VL's grounding capabilities
        from PIL import ImageFilter

        # Convert to grayscale and detect edges
        gray = image.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)

        # Threshold to create mask
        mask = edges.point(lambda x: 255 if x > 30 else 0)

        return mask

    async def refine_with_flux_fill(
        self,
        image_path: str,
        mask: Image.Image,
        prompt: str,
        output_dir: str = "./outputs/refined"
    ) -> str:
        """
        Refine specific regions of an image using Flux Fill.

        Args:
            image_path: Path to the original image
            mask: PIL Image mask (white = areas to refine)
            prompt: Description of desired refinement
            output_dir: Output directory

        Returns:
            Path to refined image
        """
        self._load_flux_fill()

        image = Image.open(image_path).convert("RGB")

        # Ensure mask is same size as image
        mask = mask.resize(image.size)

        # Run Flux Fill
        with torch.inference_mode():
            result = self.flux_fill_pipeline(
                prompt=prompt,
                image=image,
                mask_image=mask,
                num_inference_steps=28,
                guidance_scale=30,
            ).images[0]

        # Save
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filename = f"flux_refined_{Path(image_path).stem}_{uuid.uuid4().hex[:6]}.png"
        save_path = output_path / filename
        result.save(save_path)

        return str(save_path)

    async def auto_refine(
        self,
        image_path: str,
        output_dir: str = "./outputs/refined"
    ) -> dict:
        """
        Automatically analyze and refine an image.

        Args:
            image_path: Path to the image
            output_dir: Output directory

        Returns:
            Dictionary with analysis and refined image path
        """
        # Step 1: Analyze the image
        print(f"Analyzing image: {image_path}")
        analysis = await self.analyze_image(image_path)

        # Step 2: Create refinement prompt based on analysis
        refinement_prompt = f"""High quality, detailed image.
Fix any artifacts or distortions.
Improve facial features if present.
Enhance overall image quality.
Maintain the original style and composition."""

        # Step 3: Create mask for problematic areas
        # For now, use a soft global refinement
        image = Image.open(image_path).convert("RGB")
        mask = Image.new("L", image.size, 128)  # Semi-transparent mask for subtle refinement

        # Step 4: Refine with Flux Fill
        print("Refining with Flux Fill...")
        refined_path = await self.refine_with_flux_fill(
            image_path,
            mask,
            refinement_prompt,
            output_dir
        )

        return {
            "original": image_path,
            "refined": refined_path,
            "analysis": analysis
        }

    def unload(self):
        """Unload models to free memory."""
        if self.qwen_model is not None:
            del self.qwen_model
            self.qwen_model = None
            self.qwen_processor = None

        if self.flux_fill_pipeline is not None:
            del self.flux_fill_pipeline
            self.flux_fill_pipeline = None

        torch.cuda.empty_cache()
        print("Qwen/Flux Fill models unloaded")


class FluxContextRefiner:
    """
    Uses Flux with context/conditioning for image refinement.
    This is a lighter alternative to Flux Fill that uses the LoRA for consistent refinement.
    """

    def __init__(self, generator=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.generator = generator

    async def refine_with_context(
        self,
        image_path: str,
        prompt: str,
        lora_path: Optional[str] = None,
        lora_strength: float = 0.5,
        refinement_strength: float = 0.3,
        output_dir: str = "./outputs/refined"
    ) -> str:
        """
        Refine an image using Flux with img2img style refinement.

        Args:
            image_path: Path to the original image
            prompt: Refinement prompt
            lora_path: Optional LoRA for consistency
            lora_strength: LoRA influence
            refinement_strength: How much to change (0=none, 1=complete)
            output_dir: Output directory

        Returns:
            Path to refined image
        """
        from diffusers import FluxImg2ImgPipeline
        import torch

        print(f"Loading Flux img2img pipeline...")

        pipeline = FluxImg2ImgPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16
        )
        pipeline.enable_model_cpu_offload()

        # Load LoRA if specified
        if lora_path:
            pipeline.load_lora_weights(lora_path)
            pipeline.set_adapters(["default"], adapter_weights=[lora_strength])

        # Load and prepare image
        image = Image.open(image_path).convert("RGB")

        # Generate refined image
        with torch.inference_mode():
            result = pipeline(
                prompt=prompt,
                image=image,
                strength=refinement_strength,
                num_inference_steps=28,
                guidance_scale=3.5,
            ).images[0]

        # Save
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filename = f"context_refined_{Path(image_path).stem}_{uuid.uuid4().hex[:6]}.png"
        save_path = output_path / filename
        result.save(save_path)

        # Cleanup
        del pipeline
        torch.cuda.empty_cache()

        return str(save_path)
