"""Encode aligned scan previews and transparent overlay layers."""
from io import BytesIO

from PIL import Image


def png(image):
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def heatmap(gray, size, color=(248, 126, 35)):
    mask = gray.resize(size, Image.Resampling.BILINEAR)
    overlay = Image.new("RGBA", size, color)
    overlay.putalpha(mask.point(lambda value: round(value * 0.82)))
    return png(overlay)
