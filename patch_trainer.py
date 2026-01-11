"""
Patch trainer.py to add extensive debugging
"""
import re

trainer_path = r"C:\Users\benno\Documents\LoraMachine\backend\training\trainer.py"

with open(trainer_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Patch 1: Add debug output at start of training loop
content = content.replace(
    "        # Training loop\n        step = 0\n        losses = []",
    """        # Training loop
        step = 0
        losses = []

        print("\\n" + "="*70)
        print("STARTING TRAINING LOOP")
        print(f"Total steps: {steps}, Images: {len(data)}")
        print("="*70 + "\\n")"""
)

# Patch 2: Add debug output before each major operation in the loop
content = content.replace(
    "                try:\n                    # Load and preprocess image\n                    img = Image.open(item[\"image_path\"]).convert(\"RGB\")",
    """                try:
                    print(f"\\n{'='*70}")
                    print(f"STEP {step+1}/{steps} - {Path(item['image_path']).name}")
                    print(f"{'='*70}")
                    print(f"[1/10] Loading image...")

                    # Load and preprocess image
                    img = Image.open(item["image_path"]).convert("RGB")"""
)

# Patch 3: Add debug after image transform
content = content.replace(
    "                    img_tensor = transform(img).unsqueeze(0).to(self.device, dtype=torch.bfloat16)\n\n                    # Encode image to latent space",
    """                    print(f"[2/10] Transforming image...")
                    img_tensor = transform(img).unsqueeze(0).to(self.device, dtype=torch.bfloat16)
                    print(f"        Shape: {img_tensor.shape}")
                    torch.cuda.synchronize()

                    # Encode image to latent space
                    print(f"[3/10] Encoding to latent space...")"""
)

# Patch 4: Add debug after VAE encode
content = content.replace(
    "                    with torch.no_grad():\n                        latents = vae.encode(img_tensor).latent_dist.sample()\n                        latents = latents * vae.config.scaling_factor\n\n                    # Encode text prompt",
    """                    with torch.no_grad():
                        latents = vae.encode(img_tensor).latent_dist.sample()
                        latents = latents * vae.config.scaling_factor
                    print(f"        Latent shape: {latents.shape}")
                    print(f"        VRAM: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
                    torch.cuda.synchronize()

                    # Encode text prompt
                    print(f"[4/10] Encoding caption: '{item['caption'][:40]}...'")"""
)

# Patch 5: Add debug after text encoding
content = content.replace(
    "                    pooled_projections, encoder_hidden_states = encode_prompt(item[\"caption\"])\n\n                    # Create random noise",
    """                    pooled_projections, encoder_hidden_states = encode_prompt(item["caption"])
                    print(f"        Done. Pooled: {pooled_projections.shape}, Hidden: {encoder_hidden_states.shape}")
                    torch.cuda.synchronize()

                    # Create random noise
                    print(f"[5/10] Creating noise...")"""
)

# Patch 6: Add debug before packing
content = content.replace(
    "                    # Pack latents and target for Flux transformer\n                    packed_noisy_latents = pack_latents(noisy_latents, bsz, num_channels, height, width)",
    """                    # Pack latents and target for Flux transformer
                    print(f"[6/10] Packing latents...")
                    packed_noisy_latents = pack_latents(noisy_latents, bsz, num_channels, height, width)"""
)

# Patch 7: Add debug before position IDs
content = content.replace(
    "                    # Generate image and text position IDs\n                    img_ids = prepare_latent_image_ids(bsz, height // 2, width // 2, self.device, torch.bfloat16)",
    """                    # Generate image and text position IDs
                    print(f"[7/10] Generating position IDs...")
                    img_ids = prepare_latent_image_ids(bsz, height // 2, width // 2, self.device, torch.bfloat16)"""
)

# Patch 8: Add debug before forward pass
content = content.replace(
    "                    # Forward pass through transformer\n                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):\n                        # Prepare timestep conditioning",
    """                    # Forward pass through transformer
                    print(f"[8/10] FORWARD PASS - VRAM: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        # Prepare timestep conditioning"""
)

# Patch 9: Add debug right before transformer.forward call
content = content.replace(
    "                        output = transformer.forward(\n                            hidden_states=packed_noisy_latents,",
    """                        print(f"        Calling transformer.forward()...")
                        print(f"          hidden_states: {packed_noisy_latents.shape}")
                        print(f"          timestep: {timestep_cond.shape}")
                        print(f"          guidance: {guidance}")

                        output = transformer.forward(
                            hidden_states=packed_noisy_latents,"""
)

# Patch 10: Add debug after forward pass
old_pattern = r"(\s+)output = transformer\.forward\(\s+hidden_states=packed_noisy_latents,\s+timestep=timestep_cond,\s+encoder_hidden_states=encoder_hidden_states,\s+pooled_projections=pooled_projections,\s+img_ids=img_ids,\s+txt_ids=txt_ids,\s+guidance=guidance,\s+return_dict=True\s+\)\n"
new_text = r"""\1output = transformer.forward(
\1    hidden_states=packed_noisy_latents,
\1    timestep=timestep_cond,
\1    encoder_hidden_states=encoder_hidden_states,
\1    pooled_projections=pooled_projections,
\1    img_ids=img_ids,
\1    txt_ids=txt_ids,
\1    guidance=guidance,
\1    return_dict=True
\1)
\1print(f"        Forward pass COMPLETE!")
\1torch.cuda.synchronize()
"""
content = re.sub(old_pattern, new_text, content)

# Patch 11: Disable gradient checkpointing temporarily for debugging
content = content.replace(
    "        # CRITICAL: Enable gradient checkpointing to fit in 16GB VRAM\n        # Without this, the 12B model won't fit with gradients\n        print(\"Enabling gradient checkpointing for memory efficiency...\")",
    "        # DEBUGGING: Temporarily disable gradient checkpointing to isolate issue\n        print(\"DEBUGGING: Gradient checkpointing DISABLED for debugging...\")"
)

content = content.replace(
    "        # Try different methods depending on model version\n        base_model = transformer.get_base_model() if hasattr(transformer, 'get_base_model') else transformer\n        if hasattr(base_model, 'enable_gradient_checkpointing'):\n            base_model.enable_gradient_checkpointing()\n        elif hasattr(base_model, 'gradient_checkpointing_enable'):\n            base_model.gradient_checkpointing_enable()\n        elif hasattr(base_model, '_set_gradient_checkpointing'):\n            base_model._set_gradient_checkpointing(True)\n        else:\n            print(\"Warning: Could not enable gradient checkpointing - training may be slow or OOM\")",
    "        # Gradient checkpointing code commented out for debugging\n        # base_model = transformer.get_base_model() if hasattr(transformer, 'get_base_model') else transformer\n        # if hasattr(base_model, 'enable_gradient_checkpointing'):\n        #     base_model.enable_gradient_checkpointing()"
)

# Write patched file
with open(trainer_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Trainer.py patched successfully!")
print("Added extensive debugging output")
print("Disabled gradient checkpointing for debugging")
