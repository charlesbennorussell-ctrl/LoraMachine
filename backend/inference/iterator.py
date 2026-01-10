from typing import Optional, Callable, List, Dict
import asyncio
import numpy as np
from .generator import FluxGenerator


class LoRAIterator:
    """
    Iterates through LoRA strength values to find optimal settings.
    Replicates OpenArt's creativity slider behavior.
    """

    def __init__(self, generator: FluxGenerator):
        self.generator = generator

    async def run_iteration(
        self,
        prompt: str,
        negative_prompt: str = "",
        lora_path: str = "",
        steps: int = 28,
        guidance_scale: float = 3.5,
        width: int = 1024,
        height: int = 1024,
        seed: Optional[int] = None,
        strength_start: float = 0.1,
        strength_end: float = 1.0,
        strength_step: float = 0.1,
        output_dir: str = "./outputs/generated",
        progress_callback: Optional[Callable] = None
    ) -> List[Dict]:
        """
        Generate images at varying LoRA strengths from start to end.
        Uses the same seed for all iterations to allow direct comparison.

        Args:
            prompt: Text prompt for generation
            negative_prompt: Negative prompt (things to avoid)
            lora_path: Path to LoRA weights directory
            steps: Number of inference steps
            guidance_scale: Classifier-free guidance scale
            width: Output image width
            height: Output image height
            seed: Random seed (same for all iterations)
            strength_start: Starting LoRA strength
            strength_end: Ending LoRA strength
            strength_step: Step size for strength values
            output_dir: Directory to save generated images
            progress_callback: Async callback for progress updates

        Returns:
            List of dictionaries with strength, path, and seed for each image
        """
        import torch

        # Generate strength values
        strengths = np.arange(strength_start, strength_end + strength_step / 2, strength_step)
        results = []

        # Use same seed for all iterations for fair comparison
        if seed is None:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()

        print(f"\n{'='*60}")
        print(f"Starting LoRA strength iteration sweep")
        print(f"{'='*60}")
        print(f"Prompt: {prompt[:80]}...")
        print(f"LoRA: {lora_path}")
        print(f"Strengths: {strength_start:.1f} -> {strength_end:.1f} (step {strength_step:.1f})")
        print(f"Seed: {seed} (fixed for comparison)")
        print(f"{'='*60}\n")

        for i, strength in enumerate(strengths):
            strength = round(float(strength), 2)

            print(f"\n[{i+1}/{len(strengths)}] Generating with LoRA strength: {strength}")

            try:
                image_path = await self.generator.generate(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    lora_path=lora_path,
                    lora_strength=strength,
                    steps=steps,
                    guidance_scale=guidance_scale,
                    width=width,
                    height=height,
                    seed=seed,
                    output_dir=output_dir
                )

                result = {
                    "strength": strength,
                    "path": image_path,
                    "seed": seed,
                    "success": True
                }
                results.append(result)

                if progress_callback:
                    await progress_callback(strength, image_path)

            except Exception as e:
                print(f"Error at strength {strength}: {e}")
                results.append({
                    "strength": strength,
                    "path": None,
                    "seed": seed,
                    "success": False,
                    "error": str(e)
                })

            # Small delay to prevent GPU overload and allow UI updates
            await asyncio.sleep(0.2)

        print(f"\n{'='*60}")
        print(f"Iteration complete! Generated {len([r for r in results if r['success']])} images")
        print(f"{'='*60}\n")

        return results

    async def run_grid_search(
        self,
        prompt: str,
        lora_path: str,
        strength_values: List[float],
        guidance_values: List[float],
        seed: Optional[int] = None,
        output_dir: str = "./outputs/generated",
        progress_callback: Optional[Callable] = None,
        **kwargs
    ) -> List[Dict]:
        """
        Run a grid search over LoRA strength and guidance scale.
        Useful for finding optimal parameter combinations.

        Args:
            prompt: Text prompt for generation
            lora_path: Path to LoRA weights
            strength_values: List of LoRA strengths to try
            guidance_values: List of guidance scales to try
            seed: Random seed
            output_dir: Output directory
            progress_callback: Progress callback
            **kwargs: Additional generation parameters

        Returns:
            List of results for each parameter combination
        """
        import torch

        if seed is None:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()

        results = []
        total = len(strength_values) * len(guidance_values)
        current = 0

        for strength in strength_values:
            for guidance in guidance_values:
                current += 1
                print(f"[{current}/{total}] Strength: {strength}, Guidance: {guidance}")

                try:
                    image_path = await self.generator.generate(
                        prompt=prompt,
                        lora_path=lora_path,
                        lora_strength=strength,
                        guidance_scale=guidance,
                        seed=seed,
                        output_dir=output_dir,
                        **kwargs
                    )

                    result = {
                        "strength": strength,
                        "guidance": guidance,
                        "path": image_path,
                        "seed": seed,
                        "success": True
                    }
                    results.append(result)

                    if progress_callback:
                        await progress_callback(current, total, image_path)

                except Exception as e:
                    results.append({
                        "strength": strength,
                        "guidance": guidance,
                        "path": None,
                        "seed": seed,
                        "success": False,
                        "error": str(e)
                    })

                await asyncio.sleep(0.2)

        return results

    async def run_prompt_variations(
        self,
        base_prompt: str,
        prompt_suffixes: List[str],
        lora_path: str,
        lora_strength: float = 0.8,
        seed: Optional[int] = None,
        output_dir: str = "./outputs/generated",
        **kwargs
    ) -> List[Dict]:
        """
        Generate images with prompt variations.
        Useful for exploring different prompting strategies.

        Args:
            base_prompt: Base prompt that will be combined with suffixes
            prompt_suffixes: List of suffix variations to try
            lora_path: Path to LoRA weights
            lora_strength: LoRA strength
            seed: Random seed
            output_dir: Output directory
            **kwargs: Additional generation parameters

        Returns:
            List of results for each prompt variation
        """
        import torch

        if seed is None:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()

        results = []

        for i, suffix in enumerate(prompt_suffixes):
            full_prompt = f"{base_prompt}, {suffix}" if suffix else base_prompt
            print(f"[{i+1}/{len(prompt_suffixes)}] Prompt: {full_prompt[:60]}...")

            try:
                image_path = await self.generator.generate(
                    prompt=full_prompt,
                    lora_path=lora_path,
                    lora_strength=lora_strength,
                    seed=seed,
                    output_dir=output_dir,
                    **kwargs
                )

                results.append({
                    "prompt": full_prompt,
                    "suffix": suffix,
                    "path": image_path,
                    "seed": seed,
                    "success": True
                })

            except Exception as e:
                results.append({
                    "prompt": full_prompt,
                    "suffix": suffix,
                    "path": None,
                    "seed": seed,
                    "success": False,
                    "error": str(e)
                })

            await asyncio.sleep(0.2)

        return results


class IterationAnalyzer:
    """
    Analyzes iteration results to recommend optimal settings.
    """

    @staticmethod
    def analyze_results(results: List[Dict]) -> Dict:
        """
        Analyze iteration results and provide recommendations.

        Args:
            results: List of iteration results

        Returns:
            Analysis dictionary with recommendations
        """
        successful = [r for r in results if r.get("success", False)]

        if not successful:
            return {
                "status": "error",
                "message": "No successful generations to analyze"
            }

        # Group by strength if available
        by_strength = {}
        for r in successful:
            strength = r.get("strength", 0)
            by_strength.setdefault(strength, []).append(r)

        # Basic analysis
        strengths = sorted(by_strength.keys())

        return {
            "status": "success",
            "total_generated": len(successful),
            "strength_range": {
                "min": min(strengths),
                "max": max(strengths)
            },
            "recommendations": {
                "portrait": {
                    "strength": 0.7,
                    "note": "Mid-range strength often works best for portraits"
                },
                "style_transfer": {
                    "strength": 0.4,
                    "note": "Lower strength preserves more of the base model style"
                },
                "character_consistency": {
                    "strength": 0.9,
                    "note": "Higher strength for consistent character features"
                }
            },
            "seed_used": successful[0].get("seed")
        }
