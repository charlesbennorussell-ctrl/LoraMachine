"""
Debug wrapper for trainer to add extensive logging
"""
import torch
import sys
import traceback
from pathlib import Path

# Monkey-patch transformer forward to add debugging
original_forward = None

def debug_forward_wrapper(self, *args, **kwargs):
    """Wrapper around transformer.forward() with debugging"""
    print("\n" + "!"*60)
    print("TRANSFORMER.FORWARD() CALLED")
    print("!"*60)

    try:
        print(f"Args count: {len(args)}")
        print(f"Kwargs: {list(kwargs.keys())}")
        print(f"VRAM before: {torch.cuda.memory_allocated() / 1e9:.2f}GB")

        # Call original
        result = original_forward(*args, **kwargs)

        print(f"VRAM after: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
        print(f"Result type: {type(result)}")
        print("TRANSFORMER.FORWARD() COMPLETED SUCCESSFULLY")
        print("!"*60 + "\n")

        return result

    except Exception as e:
        print("\n" + "X"*60)
        print("TRANSFORMER.FORWARD() FAILED!")
        print("X"*60)
        print(f"Error: {e}")
        print(f"Traceback:")
        traceback.print_exc()
        print("X"*60 + "\n")
        raise

def patch_transformer(transformer):
    """Patch transformer to add debugging"""
    global original_forward
    original_forward = transformer.forward
    transformer.forward = lambda *args, **kwargs: debug_forward_wrapper(transformer, *args, **kwargs)
    print("[DEBUG] Transformer patched with debug wrapper")
