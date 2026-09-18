"""
utils.py — Shared helpers for SatQuery AI

Handles:
  * Robust image loading (regular RGB + optional multispectral GeoTIFF via rasterio)
  * Tiling of large satellite scenes (real scenes are often 5000x5000+ px — far
    bigger than any VLM's input resolution — so we split, run inference per tile,
    then stitch results back together)
  * Drawing bounding boxes / heatmaps for the UI
"""

import io
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False


def load_image(file_obj_or_path):
    """
    Loads an image from a path or a file-like object.
    Supports standard formats (jpg/png) via PIL, and GeoTIFF (multi-band,
    e.g. RGB + NIR) via rasterio if installed.

    Returns:
        rgb_image (PIL.Image): RGB image for display / vision models
        extra_bands (np.ndarray or None): any bands beyond RGB (e.g. NIR), for NDVI
    """
    name = getattr(file_obj_or_path, "name", str(file_obj_or_path))

    if name.lower().endswith((".tif", ".tiff")) and RASTERIO_AVAILABLE:
        with rasterio.open(file_obj_or_path) as src:
            arr = src.read()  # shape: (bands, H, W)
        arr = np.transpose(arr, (1, 2, 0))  # -> (H, W, bands)
        arr = arr.astype(np.float32)

        if arr.shape[2] >= 3:
            rgb = arr[:, :, :3]
            rgb = (255 * (rgb - rgb.min()) / (np.ptp(rgb) + 1e-6)).astype(np.uint8)
            rgb_image = Image.fromarray(rgb)
        else:
            rgb_image = Image.fromarray(arr[:, :, 0].astype(np.uint8)).convert("RGB")

        extra_bands = arr[:, :, 3:] if arr.shape[2] > 3 else None
        return rgb_image, extra_bands

    # Standard RGB path
    image = Image.open(file_obj_or_path).convert("RGB")
    return image, None


def tile_image(image: Image.Image, tile_size: int = 512, overlap: int = 64):
    """
    Splits a large image into overlapping tiles so it can be fed through
    VLMs that expect small fixed-size inputs (CLIP/BLIP ~224-384px).

    Returns a list of dicts: {"tile": PIL.Image, "x": int, "y": int}
    where (x, y) is the tile's top-left offset in the original image.
    """
    w, h = image.size
    stride = tile_size - overlap
    tiles = []

    if w <= tile_size and h <= tile_size:
        return [{"tile": image, "x": 0, "y": 0}]

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            box = (x, y, min(x + tile_size, w), min(y + tile_size, h))
            tile = image.crop(box)
            tiles.append({"tile": tile, "x": x, "y": y})
            if x + tile_size >= w:
                break
        if y + tile_size >= h:
            break
    return tiles


def offset_detections(detections, x_off, y_off):
    """Shift a tile's bounding boxes back into full-image coordinates."""
    out = []
    for d in detections:
        box = d["box"]
        out.append({
            **d,
            "box": [box[0] + x_off, box[1] + y_off, box[2] + x_off, box[3] + y_off]
        })
    return out


def draw_detections(image: Image.Image, detections, color=(255, 60, 60)):
    """Draws bounding boxes + labels on a copy of the image."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f'{d["label"]} {d["conf"]:.2f}'
        text_bg = [x1, max(0, y1 - 16), x1 + 8 * len(label), y1]
        draw.rectangle(text_bg, fill=color)
        draw.text((x1 + 2, max(0, y1 - 15)), label, fill=(255, 255, 255))
    return img


def draw_evidence_regions(image: Image.Image, regions, color=(30, 180, 255)):
    """Draws change/evidence regions on a copy of the image."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for index, region in enumerate(regions, start=1):
        x1, y1, x2, y2 = region["box"]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        label = region.get("label", f"Evidence {index}")
        text_bg = [x1, max(0, y1 - 18), x1 + 9 * len(label), y1]
        draw.rectangle(text_bg, fill=color)
        draw.text((x1 + 2, max(0, y1 - 17)), label, fill=(255, 255, 255))
    return img


def ndvi_heatmap(ndvi_array: np.ndarray) -> Image.Image:
    """Converts an NDVI array (values -1..1) into a red-yellow-green heatmap image."""
    try:
        import matplotlib.cm as cm
        norm = (np.clip(ndvi_array, -1, 1) + 1) / 2.0
        colored = (cm.get_cmap("RdYlGn")(norm)[:, :, :3] * 255).astype(np.uint8)
        return Image.fromarray(colored)
    except ImportError:
        # Fallback without matplotlib: simple manual red->green ramp
        norm = np.clip((ndvi_array + 1) / 2.0, 0, 1)
        r = ((1 - norm) * 255).astype(np.uint8)
        g = (norm * 255).astype(np.uint8)
        b = np.zeros_like(r, dtype=np.uint8)
        return Image.fromarray(np.dstack([r, g, b]))


def pil_to_bytes(image: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()
