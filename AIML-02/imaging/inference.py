"""Run versioned checkpoints and retain reviewable, pixel-aligned output."""
import hashlib
import os
from pathlib import Path

import numpy as np
import torch

from .dicom import (
    ct_intensity, ct_series, radiograph, resize_volume, target_shape, to_gray_image,
)
from .models import (
    CLASS_VERSION, SEG_VERSION, class_tensor, classifier, classify_with_cam, resize_cxr,
    segment_volume, segmenter,
)
from .render import heatmap, png


class MissingWeights(FileNotFoundError):
    pass


class Inference:
    def __init__(self):
        self.device = torch.device(os.getenv("RADREVIEW_DEVICE", "cpu"))
        self.weights = Path(os.getenv("RADREVIEW_WEIGHTS", "weights"))
        self.loaded = {}

    def model(self, kind):
        if kind not in self.loaded:
            name, version, factory = {
                "classification": ("classifier.pt", CLASS_VERSION, classifier),
                "segmentation": ("segmenter.pt", SEG_VERSION, segmenter),
            }[kind]
            path = self.weights / name
            try:
                with path.open("rb") as checkpoint_file:
                    digest = hashlib.file_digest(checkpoint_file, "sha256").hexdigest()
                    checkpoint_file.seek(0)
                    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
            except FileNotFoundError as exc:
                raise MissingWeights(name) from exc
            if checkpoint["architecture"] != version:
                raise ValueError(f"Unexpected architecture in {name}")
            model = factory()
            model.load_state_dict(checkpoint["state_dict"])
            model.eval().to(self.device)
            self.loaded[kind] = model, version, digest
        return self.loaded[kind]

    def cxr(self, model, ds):
        pixels = radiograph(ds)
        image = class_tensor(resize_cxr(pixels))[None].to(self.device)
        probability, cam = classify_with_cam(model, image)
        scan = to_gray_image(pixels)
        return {
            "title": "Chest radiograph", "modality": "XR", "view": str(ds.ViewPosition),
            "kind": "classification", "overlay": "Grad-CAM",
            "label": "Opacity signal" if probability >= 0.5 else "Low opacity signal",
            "score_label": "Predicted class score",
            "confidence": max(probability, 1 - probability), "probability": probability,
            "frame_count": 1, "width": scan.width, "height": scan.height,
        }, [(png(scan), heatmap(to_gray_image(cam), scan.size))]

    def ct(self, model, datasets):
        raw, spacing, _ = ct_series(datasets)
        size = target_shape(raw.shape, spacing)
        volume = resize_volume(ct_intensity(raw), size, order=1)
        image = torch.from_numpy(volume)[None, None]
        probability = segment_volume(model, image, self.device)[0, 0].numpy()
        mask = resize_volume(probability >= 0.5, raw.shape, order=0)
        positive = int(mask.sum())
        frames = self.ct_frames(raw, mask)
        return {
            "title": "Chest CT", "modality": "CT", "view": "AXIAL",
            "kind": "segmentation", "overlay": "Tumor mask",
            "label": "Candidate region" if positive else "No candidate region",
            "score_label": "Mean score in candidate voxels",
            "confidence": float(probability[probability >= 0.5].mean()) if positive else None,
            "frame_count": raw.shape[0], "width": raw.shape[2], "height": raw.shape[1],
            "initial_frame": int(mask.sum(axis=(1, 2)).argmax()) if positive else raw.shape[0] // 2,
        }, frames

    @staticmethod
    def ct_frames(raw, mask):
        for plane, binary in zip(raw, mask):
            scan = to_gray_image(ct_intensity(plane))
            overlay = to_gray_image(np.asarray(binary))
            yield png(scan), heatmap(overlay, scan.size, color=(99, 230, 190))
