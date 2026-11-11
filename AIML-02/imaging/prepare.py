"""Build deterministic splits and caches from the two public datasets."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pydicom
from sklearn.model_selection import train_test_split

from .dicom import ct_intensity, is_frontal, radiograph, resize_volume, target_shape
from .models import resize_cxr


def partitions(items, seed, labels=None):
    train, rest = train_test_split(items, test_size=0.30, random_state=seed, stratify=labels)
    rest_labels = [item["target"] for item in rest] if labels is not None else None
    val, test = train_test_split(rest, test_size=0.5, random_state=seed, stratify=rest_labels)
    return {"train": train, "validation": val, "test": test}


def age_band(value):
    value = str(value)
    if len(value) != 4 or value[-1] not in "DWMY" or not value[:3].isdigit():
        return "Unknown"
    age = int(value[:3]) if value[-1] == "Y" else 0
    return "Under 40" if age < 40 else "40–64" if age < 65 else "65 and older"


def prepare_cxr(source, output, seed):
    targets = {}
    with (source / "stage_2_train_labels.csv").open(newline="") as file:
        for row in csv.DictReader(file):
            patient, target = row["patientId"], int(row["Target"])
            if target not in (0, 1) or targets.setdefault(patient, target) != target:
                raise ValueError(f"Inconsistent target for {patient}")
    items = []
    for patient, target in sorted(targets.items()):
        dicom = pydicom.dcmread(source / "stage_2_train_images" / f"{patient}.dcm")
        if not is_frontal(dicom):
            raise ValueError(f"Unsupported projection or modality for {patient}")
        path = f"{patient}.png"
        resize_cxr(radiograph(dicom)).save(output / path)
        items.append({
            "image": path, "target": target, "view": str(dicom.ViewPosition),
            "sex": str(dicom.get("PatientSex") or "Unknown"),
            "age": age_band(dicom.get("PatientAge")),
        })
    return partitions(items, seed, [item["target"] for item in items])


def load_nifti_pair(image_path, mask_path):
    import nibabel as nib

    image = nib.as_closest_canonical(nib.load(image_path))
    mask = nib.as_closest_canonical(nib.load(mask_path))
    spacing = nib.affines.voxel_sizes(image.affine)
    if (
        len(image.shape) != 3 or image.shape != mask.shape
        or not np.allclose(image.affine, mask.affine, atol=1e-3)
        or not np.allclose(image.affine[:3, :3], np.diag(spacing), atol=1e-3)
    ):
        raise ValueError(f"Expected a matching axis-aligned CT/label grid: {image_path.name}")
    # Canonical RAS XYZ -> LPS ZYX, matching supported Orthanc CT series.
    pixels = image.get_fdata(dtype=np.float32)[::-1, ::-1, :].transpose(2, 1, 0)
    labels = np.asarray(mask.dataobj)[::-1, ::-1, :].transpose(2, 1, 0)
    if not np.isfinite(pixels).all() or not np.isin(labels, (0, 1)).all():
        raise ValueError(f"Invalid intensity or non-binary label: {image_path.name}")
    shape = target_shape(pixels.shape, spacing[::-1])
    return (
        resize_volume(ct_intensity(pixels), shape, order=1).astype(np.float16),
        resize_volume(labels.astype(bool), shape, order=0),
    )


def prepare_ct(source, output, seed):
    cases = sorted((source / "imagesTr").glob("lung_*.nii.gz"))
    if not cases:
        raise FileNotFoundError(f"No Task06_Lung training cases under {source}")
    items = []
    for path in cases:
        pixels, mask = load_nifti_pair(path, source / "labelsTr" / path.name)
        key = path.name.removesuffix(".nii.gz")
        image_name, mask_name = f"{key}.npy", f"{key}-mask.npy"
        np.save(output / image_name, pixels, allow_pickle=False)
        np.save(output / mask_name, mask, allow_pickle=False)
        items.append({"image": image_name, "mask": mask_name})
    return partitions(items, seed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["cxr", "ct"])
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    prepare = prepare_cxr if args.task == "cxr" else prepare_ct
    split = prepare(args.source, args.output, args.seed)
    (args.output / "split.json").write_text(json.dumps(split, indent=2) + "\n")
    print({name: len(items) for name, items in split.items()})


if __name__ == "__main__":
    main()
