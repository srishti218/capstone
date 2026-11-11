"""Decode supported DICOM pixels and put axial CT in a common ZYX/LPS grid."""
from io import BytesIO

import numpy as np
import pydicom
from PIL import Image
from pydicom.pixels import apply_modality_lut, apply_voi_lut
from scipy.ndimage import zoom


class UnsupportedStudy(ValueError):
    pass


def read_dicom(data):
    return pydicom.dcmread(BytesIO(data))


def decode_plane(ds):
    if (
        int(ds.get("NumberOfFrames", 1)) != 1
        or ds.get("PhotometricInterpretation") not in {"MONOCHROME1", "MONOCHROME2"}
    ):
        raise UnsupportedStudy("Only single-frame monochrome DICOM is supported.")
    try:
        plane = ds.pixel_array
    except (RuntimeError, ValueError, NotImplementedError, AttributeError) as exc:
        raise UnsupportedStudy("Pixel transfer syntax or encoding is unsupported.") from exc
    if plane.ndim != 2:
        raise UnsupportedStudy("Expected a single grayscale image.")
    return plane


def is_frontal(ds):
    view = str(ds.get("ViewPosition", "")).upper().replace(" ", "")
    return (
        str(ds.get("Modality")) in {"CR", "DX"}
        and view in {"AP", "PA"}
        and compatible_body_part(ds)
    )


def compatible_body_part(ds):
    body_part = str(ds.get("BodyPartExamined") or "").strip().upper()
    return body_part in {"", "CHEST", "THORAX", "LUNG"}


def radiograph(ds):
    raw = decode_plane(ds)
    padding = ds.get("PixelPaddingValue")
    valid = np.ones(raw.shape, dtype=bool)
    if padding is not None:
        other = ds.get("PixelPaddingRangeLimit", padding)
        valid = (raw < min(padding, other)) | (raw > max(padding, other))
    if not valid.any():
        raise UnsupportedStudy("Image contains only pixel padding.")
    displayed = apply_voi_lut(apply_modality_lut(raw, ds), ds).astype(np.float32)
    low, high = np.percentile(displayed[valid], [1, 99])
    if not np.isfinite([low, high]).all() or high <= low:
        raise UnsupportedStudy("Radiograph has no usable intensity range.")
    displayed = np.clip((displayed - low) / (high - low), 0, 1)
    if ds.PhotometricInterpretation == "MONOCHROME1":
        displayed = 1 - displayed
    displayed[~valid] = 0
    return displayed


def ct_series(slices):
    """Validate a regular, untilted axial stack and sort by patient position."""
    if len(slices) < 8:
        raise UnsupportedStudy("CT needs at least eight axial slices.")
    try:
        orientation = np.array([ds.ImageOrientationPatient for ds in slices], dtype=float)
        position = np.array([ds.ImagePositionPatient for ds in slices], dtype=float)
        spacing = np.array([ds.PixelSpacing for ds in slices], dtype=float)
        frames = {str(ds.FrameOfReferenceUID) for ds in slices}
        dimensions = {(int(ds.Rows), int(ds.Columns)) for ds in slices}
    except (AttributeError, TypeError, ValueError) as exc:
        raise UnsupportedStudy("CT geometry tags are missing or invalid.") from exc
    if (
        len(frames) != 1 or len(dimensions) != 1
        or orientation.shape != (len(slices), 6)
        or position.shape != (len(slices), 3) or spacing.shape != (len(slices), 2)
        or not np.isfinite(position).all() or not np.isfinite(spacing).all()
        or np.any(spacing <= 0) or not np.allclose(spacing, spacing[0], atol=1e-4)
        or not np.allclose(orientation, [1, 0, 0, 0, 1, 0], atol=1e-3)
        or not np.allclose(position[:, :2], position[0, :2], atol=0.05)
        or any(
            str(ds.get("SOPClassUID")) != str(pydicom.uid.CTImageStorage)
            or "LOCALIZER" in (ds.get("ImageType") or ())
            or not compatible_body_part(ds)
            or str(ds.get("RescaleType", "HU")).upper() != "HU"
            or "RescaleSlope" not in ds or "RescaleIntercept" not in ds
            for ds in slices
        )
    ):
        raise UnsupportedStudy("CT requires a consistent, axis-aligned axial HU series.")
    order = np.argsort(position[:, 2])
    intervals = np.diff(position[order, 2])
    step = float(np.median(intervals))
    if step <= 0 or np.any(intervals <= 0) or not np.allclose(intervals, step, atol=0.05, rtol=0.01):
        raise UnsupportedStudy("CT slice spacing is irregular or duplicated.")
    if len(slices) * np.prod(next(iter(dimensions))) > 150_000_000:
        raise UnsupportedStudy("CT exceeds the prototype volume limit.")
    ordered = [slices[i] for i in order]
    pixels = np.stack([
        apply_modality_lut(decode_plane(ds), ds).astype(np.float32) for ds in ordered
    ])
    if not np.isfinite(pixels).all():
        raise UnsupportedStudy("CT has invalid HU values.")
    return pixels, (step, *spacing[0]), ordered


def ct_intensity(pixels):
    return (np.clip(pixels, -1000, 400) + 1000) / 1400


def target_shape(shape, spacing):
    target = tuple(max(1, round(size * step / 2.0)) for size, step in zip(shape, spacing))
    if np.prod(target) > 60_000_000:
        raise UnsupportedStudy("Resampled CT exceeds the prototype volume limit.")
    return target


def resize_volume(volume, shape, order):
    factors = tuple(wanted / current for current, wanted in zip(volume.shape, shape))
    return zoom(volume, factors, order=order, mode="nearest", grid_mode=True)


def to_gray_image(array):
    return Image.fromarray(np.round(np.clip(array, 0, 1) * 255).astype(np.uint8))
