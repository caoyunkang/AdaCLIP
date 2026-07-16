from pathlib import Path


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}


def _list_images(directory):
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(
        path for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _relative(path, root):
    return Path(path).relative_to(root).as_posix()


def _find_mask(root, class_name, defect_type, image_path):
    mask_dir = root / class_name / "ground_truth" / defect_type
    candidates = [
        mask_dir / f"{image_path.stem}_mask.png",
        mask_dir / f"{image_path.stem}.png",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Ground-truth mask not found for anomalous image: "
        f"image={image_path}, expected one of={candidates}"
    )


def build_mvtec_style_meta(root, class_names):
    """Build AdaCLIP metadata from class/train/test/ground_truth folders."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    info = {"train": {}, "test": {}}
    for class_name in class_names:
        class_root = root / class_name
        if not class_root.is_dir():
            raise FileNotFoundError(f"Dataset class directory does not exist: {class_root}")

        train_entries = []
        for image_path in _list_images(class_root / "train" / "good"):
            train_entries.append({
                "img_path": _relative(image_path, root),
                "mask_path": "",
                "cls_name": class_name,
                "anomaly": 0,
            })

        test_root = class_root / "test"
        if not test_root.is_dir():
            raise FileNotFoundError(f"Dataset test directory does not exist: {test_root}")

        test_entries = []
        defect_dirs = sorted(path for path in test_root.iterdir() if path.is_dir())
        for defect_dir in defect_dirs:
            is_anomaly = defect_dir.name.lower() != "good"
            for image_path in _list_images(defect_dir):
                mask_path = ""
                if is_anomaly:
                    mask_path = _relative(
                        _find_mask(root, class_name, defect_dir.name, image_path),
                        root,
                    )
                test_entries.append({
                    "img_path": _relative(image_path, root),
                    "mask_path": mask_path,
                    "cls_name": class_name,
                    "anomaly": int(is_anomaly),
                })

        if not test_entries:
            raise ValueError(f"No test images found for class: {class_name}")

        info["train"][class_name] = train_entries
        info["test"][class_name] = test_entries

    return info
