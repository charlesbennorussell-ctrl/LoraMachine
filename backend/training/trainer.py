import torch
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional
from PIL import Image
import json
import os
import queue
import threading

# Disable xformers to prevent segfaults with incompatible triton
os.environ["XFORMERS_DISABLED"] = "1"
os.environ["DIFFUSERS_NO_XFORMERS"] = "1"


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
        # Use a queue-based approach to avoid event loop deadlocks
        progress_queue = queue.Queue()
        training_done = threading.Event()
        training_error = [None]  # Use list to allow mutation from thread

        def sync_progress(p, m):
            """Thread-safe progress callback using queue"""
            progress_queue.put((p, m))

        def run_training():
            """Run training in background thread"""
            try:
                self._train_sync(
                    name, training_dir, output_dir, steps, learning_rate,
                    batch_size, resolution, trigger_word, sync_progress
                )
            except Exception as e:
                training_error[0] = e
            finally:
                training_done.set()

        # Start training in background thread
        training_thread = threading.Thread(target=run_training, daemon=True)
        training_thread.start()

        # Poll for progress updates without blocking the event loop
        while not training_done.is_set():
            # Process any queued progress updates
            while True:
                try:
                    p, m = progress_queue.get_nowait()
                    if progress_callback:
                        await progress_callback(p, m)
                except queue.Empty:
                    break

            # Short async sleep to yield control
            await asyncio.sleep(0.1)

        # Process any remaining progress updates
        while True:
            try:
                p, m = progress_queue.get_nowait()
                if progress_callback:
                    await progress_callback(p, m)
            except queue.Empty:
                break

        # Check for errors
        if training_error[0]:
            raise training_error[0]

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

        # Force disable memory-efficient attention to avoid xformers/triton segfaults
        try:
            import diffusers
            diffusers.utils.is_xformers_available = lambda: False
        except:
            pass

        # WORKAROUND: Load components individually to avoid Windows segfault
        # FluxPipeline.from_pretrained() crashes on Windows when loading all components together
        # But loading them one by one works fine
        print("Loading Flux components individually (Windows workaround)...")
        model_id = "black-forest-labs/FLUX.1-dev"

        progress_callback(8, "Loading scheduler...")
        from diffusers import FlowMatchEulerDiscreteScheduler
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_id, subfolder="scheduler", cache_dir=cache_dir
        )

        progress_callback(10, "Loading VAE...")
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch.bfloat16, cache_dir=cache_dir
        )

        progress_callback(12, "Loading CLIP text encoder...")
        from transformers import CLIPTextModel, CLIPTokenizer
        text_encoder = CLIPTextModel.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16, cache_dir=cache_dir
        )
        tokenizer = CLIPTokenizer.from_pretrained(
            model_id, subfolder="tokenizer", cache_dir=cache_dir
        )

        progress_callback(14, "Loading T5 text encoder...")
        from transformers import T5EncoderModel, T5TokenizerFast
        text_encoder_2 = T5EncoderModel.from_pretrained(
            model_id, subfolder="text_encoder_2", torch_dtype=torch.bfloat16, cache_dir=cache_dir
        )
        tokenizer_2 = T5TokenizerFast.from_pretrained(
            model_id, subfolder="tokenizer_2", cache_dir=cache_dir
        )

        progress_callback(16, "Loading transformer (largest component)...")
        from diffusers.models import FluxTransformer2DModel
        transformer_model = FluxTransformer2DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=torch.bfloat16, cache_dir=cache_dir
        )

        progress_callback(18, "Assembling pipeline...")
        pipeline = FluxPipeline(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer_model
        )
        print("Pipeline assembled successfully!")

        # Move to device step by step to avoid memory spikes
        print("Moving pipeline to GPU...")
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

        # CRITICAL: Enable gradient checkpointing to fit in 16GB VRAM
        # Without this, the 12B model won't fit with gradients
        print("Enabling gradient checkpointing for memory efficiency...")
        # Try different methods depending on model version
        base_model = transformer.get_base_model() if hasattr(transformer, 'get_base_model') else transformer
        if hasattr(base_model, 'enable_gradient_checkpointing'):
            base_model.enable_gradient_checkpointing()
        elif hasattr(base_model, 'gradient_checkpointing_enable'):
            base_model.gradient_checkpointing_enable()
        elif hasattr(base_model, '_set_gradient_checkpointing'):
            base_model._set_gradient_checkpointing(True)
        else:
            print("Warning: Could not enable gradient checkpointing - training may be slow or OOM")

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in transformer.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")

        # Clear CUDA cache before training
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print(f"VRAM before training: {torch.cuda.memory_allocated() / 1e9:.2f}GB allocated")

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

        # Helper function to encode prompts for Flux
        # Flux uses CLIP for pooled embeddings and T5 for sequence embeddings
        def encode_prompt(prompt_text):
            # Encode with CLIP for pooled projections
            text_inputs = tokenizer(
                prompt_text,
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt"
            )
            text_input_ids = text_inputs.input_ids.to(self.device)

            with torch.no_grad():
                clip_output = text_encoder(text_input_ids, output_hidden_states=False)
                # Use pooler_output for pooled_projections (shape: [batch, hidden_dim])
                pooled_projections = clip_output.pooler_output.to(dtype=torch.bfloat16)

            # Encode with T5 for encoder_hidden_states
            text_inputs_2 = tokenizer_2(
                prompt_text,
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt"
            )
            text_input_ids_2 = text_inputs_2.input_ids.to(self.device)

            with torch.no_grad():
                t5_output = text_encoder_2(text_input_ids_2)
                # Use last_hidden_state for encoder_hidden_states (shape: [batch, seq, hidden_dim])
                encoder_hidden_states = t5_output.last_hidden_state.to(dtype=torch.bfloat16)

            return pooled_projections, encoder_hidden_states

        # Helper function to prepare latent image IDs (from FluxPipeline)
        def prepare_latent_image_ids(batch_size, height, width, device, dtype):
            """Generate positional IDs for the latent image patches."""
            latent_image_ids = torch.zeros(height, width, 3, device=device, dtype=dtype)
            latent_image_ids[..., 1] = torch.arange(height, device=device)[:, None]
            latent_image_ids[..., 2] = torch.arange(width, device=device)[None, :]
            latent_image_ids = latent_image_ids.reshape(-1, 3)
            return latent_image_ids.unsqueeze(0).expand(batch_size, -1, -1)

        # Helper function to pack latents for Flux (from FluxPipeline)
        def pack_latents(latents, batch_size, num_channels, height, width):
            """Pack latents from [B, C, H, W] to [B, H*W, C*patch_size*patch_size]."""
            # Flux uses 2x2 patches
            latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
            latents = latents.permute(0, 2, 4, 1, 3, 5)
            latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels * 4)
            return latents

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
                    pooled_projections, encoder_hidden_states = encode_prompt(item["caption"])

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

                    # Get latent dimensions
                    _, num_channels, height, width = noisy_latents.shape

                    # Pack latents and target for Flux transformer
                    packed_noisy_latents = pack_latents(noisy_latents, bsz, num_channels, height, width)
                    packed_target = pack_latents(target, bsz, num_channels, height, width)

                    # Generate image and text position IDs
                    img_ids = prepare_latent_image_ids(bsz, height // 2, width // 2, self.device, torch.bfloat16)
                    txt_ids = torch.zeros(bsz, encoder_hidden_states.shape[1], 3, device=self.device, dtype=torch.bfloat16)

                    # Forward pass through transformer
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        # Prepare timestep conditioning
                        timestep_cond = t * 1000.0

                        # Prepare guidance if the model uses guidance embeddings
                        guidance = None
                        try:
                            base_config = getattr(transformer, 'config', None)
                            if base_config is None and hasattr(transformer, 'base_model'):
                                base_config = transformer.base_model.config
                            if base_config is None and hasattr(transformer, 'get_base_model'):
                                base_config = transformer.get_base_model().config
                            if base_config and hasattr(base_config, 'guidance_embeds') and base_config.guidance_embeds:
                                guidance = torch.full([bsz], 3.5, device=self.device, dtype=torch.float32)
                        except Exception:
                            pass

                        # Debug: Print shapes on first step
                        if step == 0:
                            print(f"DEBUG: packed_noisy_latents shape: {packed_noisy_latents.shape}")
                            print(f"DEBUG: timestep_cond shape: {timestep_cond.shape}")
                            print(f"DEBUG: encoder_hidden_states shape: {encoder_hidden_states.shape}")
                            print(f"DEBUG: pooled_projections shape: {pooled_projections.shape}")
                            print(f"DEBUG: img_ids shape: {img_ids.shape}")
                            print(f"DEBUG: txt_ids shape: {txt_ids.shape}")
                            print(f"DEBUG: guidance: {guidance}")

                        output = transformer.forward(
                            hidden_states=packed_noisy_latents,
                            timestep=timestep_cond,
                            encoder_hidden_states=encoder_hidden_states,
                            pooled_projections=pooled_projections,
                            img_ids=img_ids,
                            txt_ids=txt_ids,
                            guidance=guidance,
                            return_dict=True
                        )

                        # Debug output on first step
                        if step == 0:
                            print(f"DEBUG: output type: {type(output)}")
                            if output is not None:
                                print(f"DEBUG: output keys/attrs: {dir(output)[:10]}")

                        # Extract sample from output (handle both dict and tuple)
                        if output is None:
                            raise ValueError("Transformer forward() returned None!")
                        if hasattr(output, 'sample'):
                            model_pred = output.sample
                        elif isinstance(output, tuple):
                            model_pred = output[0]
                        else:
                            model_pred = output

                        if model_pred is None:
                            raise ValueError(f"model_pred is None! output type: {type(output)}")

                        # Flow matching loss (use packed target)
                        loss = torch.nn.functional.mse_loss(model_pred, packed_target)

                    optimizer.zero_grad()
                    loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)

                    optimizer.step()
                    scheduler.step()

                    losses.append(loss.item())
                    step += 1

                    # Print every step for debugging
                    print(f"Step {step}/{steps} - Loss: {loss.item():.4f}")

                    # Progress update every step (was every 10)
                    if step % 1 == 0:
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
