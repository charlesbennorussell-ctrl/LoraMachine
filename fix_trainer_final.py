"""
Final comprehensive fix for trainer
"""
import re

trainer_path = r"C:\Users\benno\Documents\LoraMachine\backend\training\trainer.py"

with open(trainer_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: Re-enable gradient checkpointing (IT'S CRITICAL!)
content = content.replace(
    "        # DEBUGGING: Temporarily disable gradient checkpointing to isolate issue\n        print(\"DEBUGGING: Gradient checkpointing DISABLED for debugging...\")\n        # Gradient checkpointing code commented out for debugging\n        # base_model = transformer.get_base_model() if hasattr(transformer, 'get_base_model') else transformer\n        # if hasattr(base_model, 'enable_gradient_checkpointing'):\n        #     base_model.enable_gradient_checkpointing()",
    """        # CRITICAL: Enable gradient checkpointing to fit in 16GB VRAM
        print("Enabling gradient checkpointing for memory efficiency...")
        base_model = transformer.get_base_model() if hasattr(transformer, 'get_base_model') else transformer
        if hasattr(base_model, 'enable_gradient_checkpointing'):
            base_model.enable_gradient_checkpointing()
        elif hasattr(base_model, 'gradient_checkpointing_enable'):
            base_model.gradient_checkpointing_enable()
        elif hasattr(base_model, '_set_gradient_checkpointing'):
            base_model._set_gradient_checkpointing(True)
        else:
            print("Warning: Could not enable gradient checkpointing")"""
)

# Fix 2: Remove batch dimension from img_ids and txt_ids (fix deprecation warning)
content = content.replace(
    "                    img_ids = prepare_latent_image_ids(bsz, height // 2, width // 2, self.device, torch.bfloat16)\n                    txt_ids = torch.zeros(bsz, encoder_hidden_states.shape[1], 3, device=self.device, dtype=torch.bfloat16)",
    """                    img_ids = prepare_latent_image_ids(bsz, height // 2, width // 2, self.device, torch.bfloat16)
                    txt_ids = torch.zeros(bsz, encoder_hidden_states.shape[1], 3, device=self.device, dtype=torch.bfloat16)
                    # Remove batch dimension to fix deprecation warning
                    img_ids = img_ids.squeeze(0)  # [4096, 3]
                    txt_ids = txt_ids.squeeze(0)  # [512, 3]"""
)

# Fix 3: Reduce resolution default to 512 to save VRAM
content = content.replace(
    "        resolution: int = 1024,",
    "        resolution: int = 512,  # Reduced to 512 for 16GB VRAM"
)

# Fix 4: Add explicit CUDA memory management
content = content.replace(
    "        # Clear CUDA cache before training\n        import gc\n        gc.collect()\n        torch.cuda.empty_cache()\n        print(f\"VRAM before training: {torch.cuda.memory_allocated() / 1e9:.2f}GB allocated\")",
    """        # Clear CUDA cache before training
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Force aggressive memory cleanup
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        print(f"VRAM before training: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

        # Set memory allocation to be more conservative
        torch.cuda.set_per_process_memory_fraction(0.95)  # Use max 95% of VRAM"""
)

# Write fixed file
with open(trainer_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Trainer.py fixed with:")
print("  ✓ Re-enabled gradient checkpointing (CRITICAL)")
print("  ✓ Fixed txt_ids/img_ids dimensions")
print("  ✓ Reduced default resolution to 512")
print("  ✓ Added aggressive CUDA memory management")
