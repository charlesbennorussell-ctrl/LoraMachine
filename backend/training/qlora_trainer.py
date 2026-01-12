"""
QLoRA Trainer for Flux models - Optimized for 16GB VRAM.

Uses NF4 quantization (4-bit) via bitsandbytes to reduce memory from ~24GB to ~9GB.
Based on: https://huggingface.co/blog/flux-qlora

Key optimizations:
1. NF4 quantization of transformer (4-bit) - reduces ~24GB to ~6GB
2. 8-bit AdamW optimizer - reduces optimizer memory by 75%
3. Gradient checkpointing - reduces activation memory by 30-50%
4. Pre-computed latents and text embeddings - removes VAE/CLIP/T5 from GPU during training
5. BF16 mixed precision for LoRA parameters
"""

import torch
from pathlib import Path
import asyncio
from typing import Callable, Optional
from PIL import Image
import json
import os
import queue
import threading
import gc
import sys
import time
from datetime import datetime

# Disable xformers to prevent segfaults
os.environ["XFORMERS_DISABLED"] = "1"
os.environ["DIFFUSERS_NO_XFORMERS"] = "1"

# Log file for real-time monitoring
LOG_FILE = Path(__file__).parent.parent.parent / "training.log"

def log(msg: str, flush: bool = True):
    """Write to both console and log file with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=flush)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            if flush:
                f.flush()
    except:
        pass


class QLoRATrainer:
    """
    QLoRA Trainer for Flux models using NF4 quantization.

    Memory usage on RTX 4080 16GB:
    - Transformer (NF4): ~6GB
    - LoRA parameters (BF16): ~0.2GB
    - Optimizer states (8-bit): ~0.3GB
    - Activations with checkpointing: ~2-3GB
    - Total: ~9-10GB (leaves headroom)
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cache_dir = "D:/CTRL_ITERATION/flux-cache"

    async def train(
        self,
        name: str,
        training_dir: str,
        output_dir: str,
        steps: int = 1000,
        learning_rate: float = 1e-4,
        batch_size: int = 1,
        resolution: int = 512,
        trigger_word: str = "ohwx",
        lora_rank: int = 16,
        progress_callback: Optional[Callable] = None
    ):
        """
        Train a LoRA on Flux using QLoRA (4-bit quantization).

        This method uses a queue-based approach to report progress without
        blocking the async event loop.
        """
        # Clear log file at start
        try:
            with open(LOG_FILE, "w") as f:
                f.write(f"=== Training Started: {datetime.now()} ===\n")
        except:
            pass

        log(f"Starting QLoRA training: {name}")
        log(f"Steps: {steps}, LR: {learning_rate}, Rank: {lora_rank}")

        progress_queue = queue.Queue()
        training_done = threading.Event()
        training_error = [None]
        last_progress = [0, "Initializing..."]

        def sync_progress(p, m):
            progress_queue.put((p, m))
            last_progress[0] = p
            last_progress[1] = m
            log(f"PROGRESS: {p}% - {m}")

        def run_training():
            try:
                self._train_sync(
                    name, training_dir, output_dir, steps, learning_rate,
                    batch_size, resolution, trigger_word, lora_rank, sync_progress
                )
            except Exception as e:
                import traceback
                error_msg = traceback.format_exc()
                log(f"TRAINING ERROR: {error_msg}")
                training_error[0] = e
            finally:
                training_done.set()
                log("Training thread finished")

        # Start training in background thread
        training_thread = threading.Thread(target=run_training, daemon=True)
        training_thread.start()
        log("Training thread started")

        # Poll for progress updates with shorter interval
        poll_count = 0
        while not training_done.is_set():
            # Process all queued updates
            updates_processed = 0
            while True:
                try:
                    p, m = progress_queue.get_nowait()
                    if progress_callback:
                        await progress_callback(p, m)
                    updates_processed += 1
                except queue.Empty:
                    break

            # Short sleep to allow event loop to process other tasks
            await asyncio.sleep(0.05)
            poll_count += 1

            # Every 100 polls (~5 sec), log that we're still alive
            if poll_count % 100 == 0:
                log(f"[Heartbeat] Still training... Last: {last_progress[0]}% - {last_progress[1]}")

        # Process remaining updates
        while True:
            try:
                p, m = progress_queue.get_nowait()
                if progress_callback:
                    await progress_callback(p, m)
            except queue.Empty:
                break

        if training_error[0]:
            log(f"Training failed with error: {training_error[0]}")
            raise training_error[0]

        if progress_callback:
            await progress_callback(100, "Training complete!")

        log("=== Training Complete ===")

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
        lora_rank: int,
        progress_callback: Callable
    ):
        """Synchronous training implementation with QLoRA."""

        log("=== _train_sync started ===")
        progress_callback(2, "Importing libraries...")

        log("Importing transformers, diffusers, peft...")
        from transformers import BitsAndBytesConfig, CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
        from diffusers import FluxPipeline, AutoencoderKL, FlowMatchEulerDiscreteScheduler
        from diffusers.models import FluxTransformer2DModel
        from peft import LoraConfig, get_peft_model
        import bitsandbytes as bnb
        import torchvision.transforms as transforms
        log("Imports complete")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        log(f"Output path: {output_path}")

        # Phase 1: Pre-compute embeddings (VAE and text encoders on GPU temporarily)
        progress_callback(5, "Phase 1: Pre-computing embeddings...")

        training_path = Path(training_dir)
        image_files = (
            list(training_path.glob("*.png")) +
            list(training_path.glob("*.jpg")) +
            list(training_path.glob("*.jpeg")) +
            list(training_path.glob("*.webp"))
        )

        if len(image_files) == 0:
            raise ValueError(f"No training images found in {training_dir}")

        log(f"Found {len(image_files)} training images")

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
                caption = f"{trigger_word} {img_path.stem.replace('_', ' ').replace('-', ' ')}"
            data.append({"image_path": str(img_path), "caption": caption})
            print(f"  - {img_path.name}: '{caption[:50]}...'")

        # Pre-compute latents using VAE
        progress_callback(8, "Loading VAE for latent caching...")

        model_id = "black-forest-labs/FLUX.1-dev"

        vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae",
            torch_dtype=torch.bfloat16,
            cache_dir=self.cache_dir
        ).to(self.device)
        vae.eval()

        progress_callback(12, "Encoding images to latents...")
        latents_cache = []

        for i, item in enumerate(data):
            img = Image.open(item["image_path"]).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(self.device, dtype=torch.bfloat16)

            with torch.no_grad():
                latent = vae.encode(img_tensor).latent_dist.sample()
                latent = latent * vae.config.scaling_factor
                latents_cache.append(latent.cpu())  # Move to CPU to free GPU

            if (i + 1) % 5 == 0:
                progress_callback(12 + int(6 * (i + 1) / len(data)), f"Encoded {i + 1}/{len(data)} images...")

        # Free VAE memory
        del vae
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Latents cached. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f}GB")

        # Pre-compute text embeddings
        progress_callback(18, "Loading text encoders for embedding caching...")

        # Load CLIP
        text_encoder = CLIPTextModel.from_pretrained(
            model_id, subfolder="text_encoder",
            torch_dtype=torch.bfloat16, cache_dir=self.cache_dir
        ).to(self.device)
        tokenizer = CLIPTokenizer.from_pretrained(
            model_id, subfolder="tokenizer", cache_dir=self.cache_dir
        )

        # Load T5
        text_encoder_2 = T5EncoderModel.from_pretrained(
            model_id, subfolder="text_encoder_2",
            torch_dtype=torch.bfloat16, cache_dir=self.cache_dir
        ).to(self.device)
        tokenizer_2 = T5TokenizerFast.from_pretrained(
            model_id, subfolder="tokenizer_2", cache_dir=self.cache_dir
        )

        text_encoder.eval()
        text_encoder_2.eval()

        progress_callback(22, "Encoding text prompts...")
        embeddings_cache = []

        for i, item in enumerate(data):
            # Encode with CLIP
            text_inputs = tokenizer(
                item["caption"],
                padding="max_length",
                max_length=77,
                truncation=True,
                return_tensors="pt"
            )

            with torch.no_grad():
                clip_output = text_encoder(text_inputs.input_ids.to(self.device))
                pooled_projections = clip_output.pooler_output.cpu()

            # Encode with T5
            text_inputs_2 = tokenizer_2(
                item["caption"],
                padding="max_length",
                max_length=512,
                truncation=True,
                return_tensors="pt"
            )

            with torch.no_grad():
                t5_output = text_encoder_2(text_inputs_2.input_ids.to(self.device))
                encoder_hidden_states = t5_output.last_hidden_state.cpu()

            embeddings_cache.append({
                "pooled_projections": pooled_projections,
                "encoder_hidden_states": encoder_hidden_states
            })

        # Free text encoder memory
        del text_encoder, text_encoder_2, tokenizer, tokenizer_2
        gc.collect()
        torch.cuda.empty_cache()
        print(f"Embeddings cached. VRAM: {torch.cuda.memory_allocated() / 1e9:.2f}GB")

        # Phase 2: Load quantized transformer
        progress_callback(25, "Phase 2: Loading quantized transformer (NF4)...")

        # NF4 quantization config - reduces 24GB model to ~6GB
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,  # Extra compression
        )

        transformer = FluxTransformer2DModel.from_pretrained(
            model_id, subfolder="transformer",
            quantization_config=nf4_config,
            torch_dtype=torch.bfloat16,
            cache_dir=self.cache_dir
        )

        print(f"Transformer loaded (NF4). VRAM: {torch.cuda.memory_allocated() / 1e9:.2f}GB")

        # Prepare for k-bit training (custom implementation for diffusion models)
        progress_callback(30, "Preparing model for QLoRA training...")

        # Note: prepare_model_for_kbit_training is designed for LLMs with embeddings
        # For diffusion models, we manually enable gradient checkpointing and set requires_grad
        transformer.requires_grad_(False)  # Freeze base model

        # Enable gradient checkpointing manually
        if hasattr(transformer, 'enable_gradient_checkpointing'):
            transformer.enable_gradient_checkpointing()
        elif hasattr(transformer, 'gradient_checkpointing_enable'):
            transformer.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")

        # Configure LoRA
        progress_callback(32, "Configuring LoRA adapters...")

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            lora_dropout=0.0,
        )

        transformer.add_adapter(lora_config)

        # Count parameters
        trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in transformer.parameters())
        print(f"Trainable: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.4f}%)")
        print(f"VRAM after LoRA setup: {torch.cuda.memory_allocated() / 1e9:.2f}GB")

        # 8-bit AdamW optimizer
        progress_callback(35, "Setting up 8-bit optimizer...")

        optimizer = bnb.optim.AdamW8bit(
            transformer.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

        # Learning rate scheduler
        from torch.optim.lr_scheduler import CosineAnnealingLR
        scheduler = CosineAnnealingLR(optimizer, T_max=steps, eta_min=learning_rate * 0.1)

        # Helper functions for Flux
        def prepare_latent_image_ids(batch_size, height, width, device, dtype):
            latent_image_ids = torch.zeros(height, width, 3, device=device, dtype=dtype)
            latent_image_ids[..., 1] = torch.arange(height, device=device)[:, None]
            latent_image_ids[..., 2] = torch.arange(width, device=device)[None, :]
            latent_image_ids = latent_image_ids.reshape(-1, 3)
            return latent_image_ids

        def pack_latents(latents, batch_size, num_channels, height, width):
            latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
            latents = latents.permute(0, 2, 4, 1, 3, 5)
            latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels * 4)
            return latents

        # Training loop
        progress_callback(38, f"Starting training: {steps} steps on {len(data)} images...")

        step = 0
        losses = []
        start_time = time.time()

        log("")
        log("=" * 70)
        log(f"STARTING QLORA TRAINING - {steps} steps")
        log(f"VRAM at start: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
        log("=" * 70)
        log("")

        transformer.train()

        while step < steps:
            for idx, item in enumerate(data):
                if step >= steps:
                    break

                try:
                    # Get cached data
                    latents = latents_cache[idx].to(self.device, dtype=torch.bfloat16)
                    pooled_projections = embeddings_cache[idx]["pooled_projections"].to(self.device, dtype=torch.bfloat16)
                    encoder_hidden_states = embeddings_cache[idx]["encoder_hidden_states"].to(self.device, dtype=torch.bfloat16)

                    # Create noise
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]

                    # Flow matching: sample timestep
                    t = torch.rand(bsz, device=self.device)
                    t_expanded = t.view(bsz, 1, 1, 1)

                    # Interpolate
                    noisy_latents = (1 - t_expanded) * noise + t_expanded * latents
                    target = latents - noise

                    # Get dimensions
                    _, num_channels, height, width = noisy_latents.shape

                    # Pack for Flux
                    packed_noisy_latents = pack_latents(noisy_latents, bsz, num_channels, height, width)
                    packed_target = pack_latents(target, bsz, num_channels, height, width)

                    # Position IDs
                    img_ids = prepare_latent_image_ids(bsz, height // 2, width // 2, self.device, torch.bfloat16)
                    txt_ids = torch.zeros(encoder_hidden_states.shape[1], 3, device=self.device, dtype=torch.bfloat16)

                    # Timestep conditioning
                    timestep_cond = t * 1000.0

                    # Guidance (if model uses it)
                    guidance = None
                    try:
                        if hasattr(transformer, 'config') and hasattr(transformer.config, 'guidance_embeds'):
                            if transformer.config.guidance_embeds:
                                guidance = torch.full([bsz], 3.5, device=self.device, dtype=torch.float32)
                    except:
                        pass

                    # Forward pass
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        output = transformer(
                            hidden_states=packed_noisy_latents,
                            timestep=timestep_cond,
                            encoder_hidden_states=encoder_hidden_states,
                            pooled_projections=pooled_projections,
                            img_ids=img_ids,
                            txt_ids=txt_ids,
                            guidance=guidance,
                            return_dict=True
                        )

                        model_pred = output.sample if hasattr(output, 'sample') else output[0]
                        loss = torch.nn.functional.mse_loss(model_pred, packed_target)

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(transformer.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()

                    losses.append(loss.item())
                    step += 1

                    # Progress updates - every 5 steps for more responsive feedback
                    if step % 5 == 0 or step == 1:
                        avg_loss = sum(losses[-10:]) / min(len(losses), 10)
                        vram = torch.cuda.memory_allocated() / 1e9
                        elapsed = time.time() - start_time
                        steps_per_sec = step / elapsed if elapsed > 0 else 0
                        eta_sec = (steps - step) / steps_per_sec if steps_per_sec > 0 else 0
                        eta_min = eta_sec / 60

                        progress = 38 + int((step / steps) * 55)
                        msg = f"Step {step}/{steps} | Loss: {avg_loss:.4f} | {steps_per_sec:.2f} it/s | ETA: {eta_min:.1f}m"
                        progress_callback(progress, msg)
                        log(f"STEP {step}/{steps} | Loss: {loss.item():.4f} | Avg: {avg_loss:.4f} | VRAM: {vram:.2f}GB | {steps_per_sec:.2f} it/s | ETA: {eta_min:.1f}m")

                except torch.cuda.OutOfMemoryError as e:
                    log(f"OOM at step {step}! Attempting recovery...")
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue
                except Exception as e:
                    log(f"ERROR at step {step}: {e}")
                    import traceback
                    log(traceback.format_exc())
                    continue

        # Training complete stats
        total_time = time.time() - start_time
        log("")
        log("=" * 70)
        log(f"TRAINING COMPLETE!")
        log(f"Total time: {total_time/60:.1f} minutes")
        log(f"Final avg loss: {sum(losses[-10:]) / min(len(losses), 10):.4f}")
        log("=" * 70)

        # Save LoRA weights
        progress_callback(95, "Saving LoRA weights...")
        log("Saving LoRA weights...")

        # Get the LoRA state dict only (not full model)
        from peft import get_peft_model_state_dict
        lora_state_dict = get_peft_model_state_dict(transformer)

        # Convert PEFT format to diffusers format
        # PEFT: "single_transformer_blocks.0.attn.to_k.lora_A.weight"
        # Diffusers: "transformer.single_transformer_blocks.0.attn.to_k.lora_A.weight"
        diffusers_state_dict = {}
        for key, value in lora_state_dict.items():
            # Add transformer prefix for diffusers compatibility
            new_key = f"transformer.{key}"
            diffusers_state_dict[new_key] = value

        # Save in diffusers-compatible format (this is the small LoRA file)
        from safetensors.torch import save_file
        save_file(diffusers_state_dict, output_path / "pytorch_lora_weights.safetensors")

        # Save adapter config for PEFT compatibility (small JSON file)
        # Skip full model save - it creates a 6GB file unnecessarily
        if hasattr(transformer, 'peft_config'):
            adapter_config = {}
            for name, peft_cfg in transformer.peft_config.items():
                cfg_dict = peft_cfg.to_dict()
                # Convert sets to lists for JSON serialization
                for k, v in cfg_dict.items():
                    if isinstance(v, set):
                        cfg_dict[k] = list(v)
                adapter_config[name] = cfg_dict
            with open(output_path / "adapter_config.json", "w") as f:
                json.dump(adapter_config.get("default", adapter_config), f, indent=2)

        # Save training config
        config = {
            "name": name,
            "trigger_word": trigger_word,
            "steps": steps,
            "learning_rate": learning_rate,
            "resolution": resolution,
            "batch_size": batch_size,
            "base_model": model_id,
            "lora_rank": lora_rank,
            "lora_alpha": lora_rank,
            "training_images": len(data),
            "final_loss": sum(losses[-10:]) / min(len(losses), 10) if losses else 0,
            "quantization": "nf4",
            "optimizer": "adamw_8bit"
        }

        with open(output_path / "training_config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Cleanup
        del transformer, optimizer
        gc.collect()
        torch.cuda.empty_cache()

        print(f"\nLoRA saved to: {output_path}")
        print(f"Final loss: {config['final_loss']:.4f}")

        return str(output_path)


class GGUFTrainer:
    """
    Alternative trainer using GGUF quantized models from ComfyUI.

    Uses your existing flux1-dev-Q8_0.gguf for inference validation.
    Training still uses NF4 via bitsandbytes (GGUF doesn't support training).
    """

    def __init__(self, gguf_path: str = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.gguf_path = gguf_path or "D:/COMFY UI/StabilityMatrix-win-x64/Data/Packages/ComfyUI/models/unet/flux1-dev-Q8_0.gguf"
        self.cache_dir = "D:/CTRL_ITERATION/flux-cache"

    def validate_lora(self, lora_path: str, prompt: str, output_path: str = None):
        """
        Validate a trained LoRA by generating an image using GGUF model.

        This uses the Q8 GGUF model for efficient inference on 16GB GPU.
        """
        from diffusers import FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig

        print(f"Loading GGUF model from: {self.gguf_path}")

        # Load GGUF transformer
        transformer = FluxTransformer2DModel.from_single_file(
            self.gguf_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            torch_dtype=torch.bfloat16,
        )

        # Create pipeline
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            transformer=transformer,
            torch_dtype=torch.bfloat16,
            cache_dir=self.cache_dir
        )

        # Enable CPU offloading (compatible with GGUF)
        pipe.enable_model_cpu_offload()

        # Load LoRA
        print(f"Loading LoRA from: {lora_path}")
        pipe.load_lora_weights(lora_path)

        # Generate
        print(f"Generating: {prompt}")
        image = pipe(
            prompt,
            height=512,
            width=512,
            guidance_scale=3.5,
            num_inference_steps=20,
            generator=torch.Generator("cpu").manual_seed(42)
        ).images[0]

        if output_path:
            image.save(output_path)
            print(f"Saved to: {output_path}")

        return image
