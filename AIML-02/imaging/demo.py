"""A clearly labeled sample review, independent of any model checkpoint."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from .render import heatmap, png


def seed_demo(store):
    result_id = "demo-cxr-001"
    if store.has_result(result_id):
        return
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "chest-pa.jpg"
    with Image.open(fixture) as source:
        scan = source.convert("L")
    width, height = scan.size
    mask = Image.new("L", scan.size)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((width * 0.26, height * 0.52, width * 0.43, height * 0.76), fill=225)
    draw.ellipse((width * 0.31, height * 0.47, width * 0.46, height * 0.65), fill=150)
    mask = mask.filter(ImageFilter.GaussianBlur(width * 0.027))
    store.save_result(result_id, {
        "title": "Chest radiograph", "modality": "XR", "view": "PA",
        "kind": "classification", "overlay": "Illustrative heatmap",
        "model": "Example DenseNet-121", "confidence": 0.874,
        "score_label": "Illustrative confidence", "label": "Opacity signal",
        "probability": 0.874, "demo": True, "frame_count": 1,
        "width": width, "height": height,
    }, [(png(scan), heatmap(mask, scan.size))])
