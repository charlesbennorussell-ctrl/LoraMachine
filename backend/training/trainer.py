import torch
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional
from PIL import Image
import json
import os


class LoRATrainer:
    """
    LoRA Trainer for Flux models.

    This implementation uses PEFT (Parameter-Efficient Fine-Tuning) for training LoRAs.
    Optimized for RTX 4080 with 16GB VRAM.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipeline = None
        self._executor = ThreadPoolExecutor(max_workers=1)

    async def train(
        self,
        name: str,
        training_dir: str,
        output_dir: str,
        steps: int = 1000,
        learning_rate: float = 1e-4,
        batch_size: int = 1,
        resolution: int = 1024,
        trigger_word: str = "ohwx",
        progress_callback: Optional[Callable] = None
    ):
        """
        Train a LoRA on Flux using PEFT library.
        Optimized for RTX 4080 16GB VRAM.

        Args:
            name: Name for the LoRA
            training_dir: Directory containing training images
            output_dir: Where to save the trained LoRA
            steps: Number of training steps
            learning_rate: Learning rate for optimization
            batch_size: Batch size (keep at 1 for 16GB VRAM)
            resolution: Training resolution
            trigger_word: Trigger word for the trained concept
            progress_callback: Async callback for progress updates
        """
        # Store the event loop for callbacks from the thread
        loop = asyncio.get_event_loop()

        def sync_progress(p, m):
            """Thread-safe progress callback that schedules on the main event loop"""
            if progress_callback:
                asyncio.run_coroutine_threadsafe(progress_callback(p, m), loop)

        # Run the blocking training in a thread pool
        await loop.run_in_executor(
            self._executor,
            self._train_sync,
            name, training_dir, output_dir, steps, learning_rate,
            batch_size, resolution, trigger_word, sync_progress
        )

        # Final callback after training completes
        if progress_callback:
            await progress_callback(100, "Training complete!")

    def _train_sync(
        self,
        name: str,
        training_dir: str,
        output_dir: str,
        steps: int,
        learning_rate: float,
        batch_size: int,
        resolution: int,
        trigger_word: str,
        progress_callback: Callable
    ):
        """
        Synchronous training implementation that runs in a thread pool.
        This keeps heavy GPU/IO operations off the async event loop.
        """
        from diffusers import FluxPipeline
        from peft import LoraConfig, get_peft_model
        import torchvision.transforms as transforms

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        progress_callback(5, "Loading Flux pipeline...")

        # Load full pipeline (more efficient than loading components separately)
        print("Loading Flux pipeline for LoRA training...")
        # Use D: drive for cache to avoid C: drive space issues
        cache_dir = "D:/CTRL_ITERATION/flux-cache"

        pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir
        )
        pipeline.to(self.device)

        # Extract components
        transformer = pipeline.transformer
        vae = pipeline.vae
        text_encoder = pipeline.text_encoder
        text_encoder_2 = pipeline.text_encoder_2
        tokenizer = pipeline.tokenizer
        tokenizer_2 = pipeline.tokenizer_2

        # Set models to eval mode (except transformer which we'll train)
        vae.eval()
        text_encoder.eval()
        text_encoder_2.eval()

        progress_callback(15, "Configuring LoRA...")

        # Configure LoRA
        lora_config = LoraConfig(
            r=16,  # LoRA rank
            lora_alpha=16,
            target_modules=[
                "to_q", "to_k", "to_v", "to_out.0",
                "proj_in", "proj_out",
                "ff.net.0.proj", "ff.net.2"
            ],
            lora_dropout=0.0,
            bias="none",
        )

        # Apply LoRA to transformer
        transformer = get_peft_model(transformer, lora_config)
        transformer.to(self.device)
        transformer.train()

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in transformer.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

        progress_callback(20, "Preparing training data...")

        # Prepare dataset
        training_path = Path(training_dir)
        image_files = (
            list(training_path.glob("*.png")) +
            list(training_path.glob("*.jpg")) +
            list(training_path.glob("*.jpeg")) +
            list(training_path.glob("*.webp"))
        )

        if len(image_files) == 0:
            raise ValueError(f"No training images found in {training_dir}")

        print(f"Found {len(image_files)} training images")

        # Image preprocessing
        transform = transforms.Compose([
            transforms.Resize((resolution, resolution), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        # Create captions
        data = []
        for img_path in image_files:
            caption_path = img_path.with_suffix(".txt")
            if caption_path.exists():
                caption = caption_path.read_text().strip()
            else:
                # Auto-generate caption with trigger word
                caption = f"{trigger_word} {img_path.stem.replace('_', ' ').replace('-', ' ')}"
            data.append({"image_path": str(img_path), "caption": caption})
            print(f"  - {img_path.name}: '{caption[:50]}...'")

        progress_callback(25, f"Starting training on {len(data)} images...")

        # Optimizer
        optimizer = torch.optim.AdamW(
            transformer.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )

        # Learning rate scheduler
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=steps, eta_min=learning_rate * 0.1)

        # Helper function to encode prompts
        def encode_prompt(prompt_text):
            # Encode with CLIP
            text_inputs = tokenizer(
                prompt_text,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt"
            )
            text_input_ids = text_inputs.input_ids.to(self.device)

            with torch.no_grad():
                prompt_embeds = text_encoder(text_input_ids)[0]

            # Encode with T5
            text_inputs_2 = tokenizer_2(
                prompt_text,
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt"
            )
            text_input_ids_2 = text_inputs_2.input_ids.to(self.device)

            with torch.no_grad():
                prompt_embeds_2 = text_encoder_2(text_input_ids_2)[0]

            return prompt_embeds, prompt_embeds_2

        # Training loop
        step = 0
        losses = []

        while step < steps:
            for item in data:
                if step >= steps:
                    break

                try:
                    # Load and preprocess image
                    img = Image.open(item["image_path"]).convert("RGB")
                    img_tensor = transform(img).unsqueeze(0).to(self.device, dtype=torch.bfloat16)

                    # Encode image to latent space
                    with torch.no_grad():
                        latents = vae.encode(img_tensor).latent_dist.sample()
                        latents = latents * vae.config.scaling_factor

                    # Encode text prompt
                    prompt_embeds, prompt_embeds_2 = encode_prompt(item["caption"])

                    # Create random noise and timestep for flow matching
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]

                    # Flow matching: interpolate between noise and data
                    # t ~ Uniform(0, 1)
                    t = torch.rand(bsz, device=self.device)
                    t_expanded = t.view(bsz, 1, 1, 1)

                    # Interpolate: x_t = (1-t) * noise + t * latents
                    noisy_latents = (1 - t_expanded) * noise + t_expanded * latents

                    # Target is the velocity: dx/dt = latents - noise
                    target = latents - noise

                    # Forward pass through transformer
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        # Prepare timestep conditioning
                        timestep_cond = t * 1000.0

                        # Run transformer
                        model_pred = transformer(
                            hidden_states=noisy_latents,
                            timestep=timestep_cond,
                            encoder_hidden_states=prompt_embeds,
                            pooled_projections=prompt_embeds_2,
                            return_dict=False
                        )[0]

                        # Flow matching loss
                        loss = torch.nn.functional.mse_loss(model_pred, target)

                    optimizer.zero_grad()
                    loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)

                    optimizer.step()
                    scheduler.step()

                    losses.append(loss.item())
                    step += 1

                    # Progress update every 10 steps
                    if step % 10 == 0:
                        avg_loss = sum(losses[-10:]) / min(len(losses), 10)
                        progress = 25 + int((step / steps) * 70)
                        progress_callback(
                            progress,
                            f"Step {step}/{steps} - Loss: {avg_loss:.4f} - LR: {scheduler.get_last_lr()[0]:.2e}"
                        )

                except Exception as e:
                    print(f"Error processing {item['image_path']}: {e}")
                    continue

        progress_callback(95, "Saving LoRA weights...")

        # Save LoRA weights
        transformer.save_pretrained(output_path)

        # Save training config
        config = {
            "name": name,
            "trigger_word": trigger_word,
            "steps": steps,
            "learning_rate": learning_rate,
            "resolution": resolution,
            "batch_size": batch_size,
            "base_model": "black-forest-labs/FLUX.1-dev",
            "lora_rank": 16,
            "lora_alpha": 16,
            "training_images": len(data),
            "final_loss": sum(losses[-10:]) / min(len(losses), 10) if losses else 0
        }

        with open(output_path / "training_config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Cleanup
        del pipeline
        torch.cuda.empty_cache()

        print(f"LoRA saved to: {output_path}")
        return str(output_path)


class SimpleTunerWrapper:
    """
    Alternative trainer using SimpleTuner for more robust Flux LoRA training.
    Use this if the PEFT-based trainer doesn't produce good results.
    """

    def __init__(self, simpletuner_path: str = None):
        self.simpletuner_path = simpletuner_path or "./SimpleTuner"

    async def train(
        self,
        name: str,
        training_dir: str,
        output_dir: str,
        steps: int = 1000,
        learning_rate: float = 1e-4,
        resolution: int = 1024,
        trigger_word: str = "ohwx",
        progress_callback: Optional[Callable] = None
    ):
        """
        Train using SimpleTuner (requires SimpleTuner to be installed).
        """
        import subprocess
        import shutil

        # Create SimpleTuner config
        config = {
            "model_type": "lora",
            "pretrained_model_name_or_path": "black-forest-labs/FLUX.1-dev",
            "instance_data_dir": training_dir,
            "output_dir": output_dir,
            "instance_prompt": trigger_word,
            "resolution": resolution,
            "train_batch_size": 1,
            "gradient_accumulation_steps": 4,
            "learning_rate": learning_rate,
            "max_train_steps": steps,
            "mixed_precision": "bf16",
            "lora_rank": 16,
        }

        config_path = Path(output_dir) / "simpletuner_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        # Run SimpleTuner
        cmd = [
            "python", f"{self.simpletuner_path}/train.py",
            "--config", str(config_path)
        ]

        if progress_callback:
            await progress_callback(10, "Starting SimpleTuner training...")

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT
        )

        while True:
            line = await process.stdout.readline()
            if not line:
                break
            line = line.decode().strip()
            print(line)

            # Parse progress from output
            if "step" in line.lower():
                try:
                    # Extract step number if present
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if p.lower() == "step":
                            current = int(parts[i + 1].replace(",", "").replace(":", ""))
                            progress = 10 + int((current / steps) * 85)
                            if progress_callback:
                                await progress_callback(progress, line[:80])
                            break
                except:
                    pass

        await process.wait()

        if progress_callback:
            await progress_callback(100, "Training complete!")

        return output_dir
