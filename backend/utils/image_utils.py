"""
Image utility functions for the Flux LoRA Pipeline.
"""

from PIL import Image
from pathlib import Path
from typing import Tuple, Dict, Optional
import hashlib


def resize_image(
    image_path: str,
    target_size: Tuple[int, int],
    output_path: Optional[str] = None,
    keep_aspect: bool = True
) -> str:
    """
    Resize an image to target size.

    Args:
        image_path: Path to input image
        target_size: Target (width, height)
        output_path: Optional output path (defaults to same as input)
        keep_aspect: Whether to maintain aspect ratio

    Returns:
        Path to resized image
    """
    img = Image.open(image_path)

    if keep_aspect:
        img.thumbnail(target_size, Image.LANCZOS)
    else:
        img = img.resize(target_size, Image.LANCZOS)

    save_path = output_path or image_path
    img.save(save_path, quality=95)
    return save_path


def create_thumbnail(
    image_path: str,
    size: Tuple[int, int] = (256, 256),
    output_dir: Optional[str] = None
) -> str:
    """
    Create a thumbnail of an image.

    Args:
        image_path: Path to input image
        size: Thumbnail size
        output_dir: Output directory (defaults to same directory)

    Returns:
        Path to thumbnail
    """
    input_path = Path(image_path)
    output_dir = Path(output_dir) if output_dir else input_path.parent

    img = Image.open(image_path)
    img.thumbnail(size, Image.LANCZOS)

    thumb_path = output_dir / f"thumb_{input_path.name}"
    img.save(thumb_path, quality=85)

    return str(thumb_path)


def get_image_info(image_path: str) -> Dict:
    """
    Get information about an image.

    Args:
        image_path: Path to image

    Returns:
        Dictionary with image information
    """
    path = Path(image_path)
    img = Image.open(image_path)

    # Calculate hash for duplicate detection
    with open(image_path, "rb") as f:
        file_hash = hashlib.md5(f.read()).hexdigest()

    return {
        "path": str(path),
        "name": path.name,
        "size_bytes": path.stat().st_size,
        "width": img.width,
        "height": img.height,
        "format": img.format,
        "mode": img.mode,
        "hash": file_hash
    }


def crop_to_square(image_path: str, output_path: Optional[str] = None) -> str:
    """
    Crop an image to a square (center crop).

    Args:
        image_path: Path to input image
        output_path: Optional output path

    Returns:
        Path to cropped image
    """
    img = Image.open(image_path)

    min_dim = min(img.width, img.height)
    left = (img.width - min_dim) // 2
    top = (img.height - min_dim) // 2
    right = left + min_dim
    bottom = top + min_dim

    cropped = img.crop((left, top, right, bottom))

    save_path = output_path or image_path
    cropped.save(save_path, quality=95)
    return save_path


def prepare_training_image(
    image_path: str,
    output_dir: str,
    target_resolution: int = 1024,
    crop_to_square_first: bool = True
) -> str:
    """
    Prepare an image for LoRA training.

    Args:
        image_path: Path to input image
        output_dir: Output directory
        target_resolution: Target resolution
        crop_to_square_first: Whether to crop to square first

    Returns:
        Path to prepared image
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    input_path = Path(image_path)
    save_path = output_path / input_path.name

    img = Image.open(image_path).convert("RGB")

    if crop_to_square_first:
        min_dim = min(img.width, img.height)
        left = (img.width - min_dim) // 2
        top = (img.height - min_dim) // 2
        img = img.crop((left, top, left + min_dim, top + min_dim))

    img = img.resize((target_resolution, target_resolution), Image.LANCZOS)
    img.save(save_path, quality=95)

    return str(save_path)


def create_image_grid(
    image_paths: list,
    grid_size: Tuple[int, int],
    cell_size: Tuple[int, int] = (256, 256),
    padding: int = 4
) -> Image.Image:
    """
    Create a grid of images.

    Args:
        image_paths: List of image paths
        grid_size: Grid dimensions (cols, rows)
        cell_size: Size of each cell
        padding: Padding between cells

    Returns:
        PIL Image of the grid
    """
    cols, rows = grid_size

    grid_width = cols * cell_size[0] + (cols - 1) * padding
    grid_height = rows * cell_size[1] + (rows - 1) * padding

    grid = Image.new("RGB", (grid_width, grid_height), (30, 30, 30))

    for i, path in enumerate(image_paths[:cols * rows]):
        row = i // cols
        col = i % cols

        try:
            img = Image.open(path)
            img.thumbnail(cell_size, Image.LANCZOS)

            x = col * (cell_size[0] + padding)
            y = row * (cell_size[1] + padding)

            # Center in cell
            x_offset = (cell_size[0] - img.width) // 2
            y_offset = (cell_size[1] - img.height) // 2

            grid.paste(img, (x + x_offset, y + y_offset))
        except Exception as e:
            print(f"Error adding image to grid: {e}")

    return grid
