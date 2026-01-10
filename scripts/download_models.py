"""
Download required models for the Flux LoRA Pipeline.
Run this script after installing dependencies.
"""

import os
import sys
from pathlib import Path
import urllib.request
import hashlib

# Get project root
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"


def download_file(url: str, output_path: Path, expected_hash: str = None):
    """Download a file with progress indicator."""
    print(f"Downloading: {url}")
    print(f"To: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"File already exists, skipping...")
        return True

    try:
        def progress_hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            print(f"\rProgress: {percent}%", end="", flush=True)

        urllib.request.urlretrieve(url, str(output_path), reporthook=progress_hook)
        print("\nDownload complete!")

        # Verify hash if provided
        if expected_hash:
            print("Verifying file integrity...")
            with open(output_path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
            if file_hash != expected_hash:
                print(f"WARNING: Hash mismatch!")
                print(f"Expected: {expected_hash}")
                print(f"Got: {file_hash}")
                return False

        return True
    except Exception as e:
        print(f"\nError downloading: {e}")
        return False


def download_realesrgan():
    """Download Real-ESRGAN x4plus model."""
    print("\n" + "=" * 50)
    print("Downloading Real-ESRGAN x4plus Model")
    print("=" * 50)

    url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    output_path = MODELS_DIR / "refiners" / "RealESRGAN_x4plus.pth"

    return download_file(url, output_path)


def download_gfpgan():
    """Download GFPGAN v1.4 model for face enhancement."""
    print("\n" + "=" * 50)
    print("Downloading GFPGAN v1.4 Model (Face Enhancement)")
    print("=" * 50)

    url = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
    output_path = MODELS_DIR / "refiners" / "GFPGANv1.4.pth"

    return download_file(url, output_path)


def check_huggingface_login():
    """Check if user is logged into HuggingFace."""
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            print("HuggingFace: Logged in")
            return True
        else:
            print("HuggingFace: Not logged in")
            return False
    except ImportError:
        print("HuggingFace Hub not installed")
        return False


def download_flux_model():
    """Download Flux model from HuggingFace."""
    print("\n" + "=" * 50)
    print("Checking Flux Model")
    print("=" * 50)

    flux_dir = MODELS_DIR / "flux"

    # Check if already downloaded
    if any(flux_dir.glob("*.safetensors")) or any(flux_dir.glob("**/*.safetensors")):
        print("Flux model appears to be already downloaded.")
        return True

    if not check_huggingface_login():
        print("\nTo download Flux, you need to:")
        print("1. Accept the license at: https://huggingface.co/black-forest-labs/FLUX.1-dev")
        print("2. Run: huggingface-cli login")
        print("\nSkipping Flux download for now. It will auto-download on first use.")
        return False

    print("\nDownloading Flux model (this may take a while - ~30GB)...")

    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            "black-forest-labs/FLUX.1-dev",
            local_dir=str(flux_dir),
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print("Flux model downloaded successfully!")
        return True
    except Exception as e:
        print(f"Error downloading Flux: {e}")
        print("The model will auto-download when first used.")
        return False


def main():
    print("=" * 50)
    print("Flux LoRA Pipeline - Model Downloader")
    print("=" * 50)

    # Create directories
    (MODELS_DIR / "flux").mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "loras").mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "refiners").mkdir(parents=True, exist_ok=True)

    results = []

    # Download refinement models
    results.append(("Real-ESRGAN", download_realesrgan()))
    results.append(("GFPGAN", download_gfpgan()))

    # Check Flux
    results.append(("Flux", download_flux_model()))

    # Summary
    print("\n" + "=" * 50)
    print("Download Summary")
    print("=" * 50)
    for name, success in results:
        status = "OK" if success else "SKIPPED/FAILED"
        print(f"  {name}: {status}")

    print("\n" + "=" * 50)
    print("Next Steps:")
    print("=" * 50)
    print("1. If Flux wasn't downloaded, accept the license at:")
    print("   https://huggingface.co/black-forest-labs/FLUX.1-dev")
    print("2. Login to HuggingFace: huggingface-cli login")
    print("3. The Flux model will auto-download on first use")
    print("=" * 50)


if __name__ == "__main__":
    main()
