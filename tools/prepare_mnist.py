import gzip
import io
import struct
import subprocess
from array import array
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


REPO_URL = "https://huggingface.co/datasets/ylecun/mnist"


def ensure_repo(root: Path) -> Path:
    """Clone the HuggingFace MNIST dataset (uses git-lfs) if not present."""

    target = root / "source"
    if (target / ".git").exists():
        print(f"Reusing existing repository at {target}")
        return target

    if target.exists():
        has_known_data = any(target.rglob("*.parquet")) or any(
            target.rglob("*idx*-ubyte.gz")
        )
        if has_known_data:
            print(
                f"Reusing existing dataset at {target} (no git metadata detected)"
            )
            return target

    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning {REPO_URL} -> {target}")
    try:
        subprocess.run(
            ["git", "clone", REPO_URL, str(target)], check=True, cwd=root
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Failed to clone MNIST repository. If you already downloaded the dataset "
            f"(e.g., via `git clone {REPO_URL}`), place it under {target} and rerun. "
            "HuggingFace parquet exports named train-00000-of-00001.parquet and "
            "test-00000-of-00001.parquet are also supported."
        ) from exc
    return target


def find_file(repo_root: Path, filename: str) -> Path:
    for path in repo_root.rglob(filename):
        if path.is_file():
            return path
    raise FileNotFoundError(f"Could not find {filename} under {repo_root}")


def find_parquet(repo_root: Path, split: str) -> Path:
    pattern = f"{split}-*.parquet"
    for path in repo_root.rglob(pattern):
        if path.is_file():
            return path
    raise FileNotFoundError(f"Could not find parquet split matching {pattern} under {repo_root}")


def _read_idx_header(raw: bytes) -> Tuple[int, int, List[int]]:
    if len(raw) < 4:
        raise ValueError("IDX file too small")
    magic = raw[0:4]
    data_type = magic[2]
    num_dims = magic[3]
    if data_type != 0x08:
        raise ValueError(f"Unsupported data type code {data_type}")
    if len(raw) < 4 + 4 * num_dims:
        raise ValueError("IDX header truncated")
    dims: List[int] = []
    offset = 4
    for _ in range(num_dims):
        dims.append(struct.unpack_from(">I", raw, offset)[0])
        offset += 4
    return data_type, num_dims, dims


def read_images(path: Path) -> Tuple[List[int], Tuple[int, ...]]:
    with gzip.open(path, "rb") as f:
        raw = f.read()
    data_type, num_dims, dims = _read_idx_header(raw)
    if num_dims != 3 or dims[1:] != [28, 28]:
        raise ValueError(f"Unexpected image shape {dims}")
    offset = 4 + 4 * num_dims
    pixel_data = raw[offset:]
    if len(pixel_data) != dims[0] * 28 * 28:
        raise ValueError("Image payload size mismatch")
    return list(pixel_data), (dims[0], 28, 28)


def read_labels(path: Path) -> Tuple[List[int], Tuple[int, ...]]:
    with gzip.open(path, "rb") as f:
        raw = f.read()
    data_type, num_dims, dims = _read_idx_header(raw)
    if num_dims != 1:
        raise ValueError(f"Unexpected label dimensions {dims}")
    offset = 4 + 4 * num_dims
    label_bytes = raw[offset:]
    if len(label_bytes) != dims[0]:
        raise ValueError("Label payload size mismatch")
    return list(label_bytes), (dims[0],)


def to_one_hot(labels: Sequence[int], num_classes: int = 10) -> array:
    result = array("f", [0.0] * (len(labels) * num_classes))
    for idx, label in enumerate(labels):
        if label < 0 or label >= num_classes:
            raise ValueError(f"Label out of range: {label}")
        result[idx * num_classes + label] = 1.0
    return result


def normalize_images(pixels: Iterable[int]) -> array:
    return array("f", (value / 255.0 for value in pixels))


def write_tensor(path: Path, shape: Sequence[int], data: array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = array("i", [len(shape), *shape])
    with open(path, "wb") as f:
        header.tofile(f)
        data.tofile(f)
    print(f"Wrote tensor {tuple(shape)} -> {path}")


def load_split_from_parquet(parquet_path: Path) -> Tuple[List[int], Tuple[int, ...], List[int], Tuple[int, ...]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "pyarrow is required to read parquet MNIST exports. Install with `pip install pyarrow`."
        ) from exc

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Pillow is required to decode PNG images embedded in the parquet file. Install with `pip install pillow`."
        ) from exc

    table = pq.read_table(parquet_path, columns=["image", "label"])
    labels = [int(value) for value in table.column("label").to_pylist()]

    pixels: List[int] = []
    for entry in table.column("image").to_pylist():
        if isinstance(entry, dict):
            blob = entry.get("bytes") or entry.get("data")
        elif isinstance(entry, (bytes, bytearray, memoryview)):
            blob = bytes(entry)
        else:
            raise ValueError(f"Unsupported image payload type: {type(entry)}")

        if blob is None:
            raise ValueError("Image entry missing bytes field")

        with Image.open(io.BytesIO(blob)) as img:
            img = img.convert("L")
            if img.size != (28, 28):
                raise ValueError(f"Unexpected image size {img.size} in {parquet_path}")
            pixels.extend(list(img.getdata()))

    image_shape = (len(labels), 28, 28)
    label_shape = (len(labels),)
    if len(pixels) != image_shape[0] * 28 * 28:
        raise ValueError("Parquet image payload size mismatch")
    return pixels, image_shape, labels, label_shape


def prepare_dataset(root: Path) -> None:
    repo_root = ensure_repo(root)

    processed_dir = root / "processed"

    try:
        train_images_path = find_file(repo_root, "train-images-idx3-ubyte.gz")
        test_images_path = find_file(repo_root, "t10k-images-idx3-ubyte.gz")
        train_labels_path = find_file(repo_root, "train-labels-idx1-ubyte.gz")
        test_labels_path = find_file(repo_root, "t10k-labels-idx1-ubyte.gz")
    except FileNotFoundError:
        print("IDX archives not found; falling back to parquet exports")
        train_parquet = find_parquet(repo_root, "train")
        test_parquet = find_parquet(repo_root, "test")

        train_pixels, train_image_shape, train_labels, train_label_shape = (
            load_split_from_parquet(train_parquet)
        )
        test_pixels, test_image_shape, test_labels, test_label_shape = (
            load_split_from_parquet(test_parquet)
        )
    else:
        train_pixels, train_image_shape = read_images(train_images_path)
        test_pixels, test_image_shape = read_images(test_images_path)
        train_labels, train_label_shape = read_labels(train_labels_path)
        test_labels, test_label_shape = read_labels(test_labels_path)

    if train_image_shape[0] != train_label_shape[0]:
        raise ValueError("Train image/label count mismatch")
    if test_image_shape[0] != test_label_shape[0]:
        raise ValueError("Test image/label count mismatch")

    train_images_flat = normalize_images(train_pixels)
    test_images_flat = normalize_images(test_pixels)
    train_labels_oh = to_one_hot(train_labels)
    test_labels_oh = to_one_hot(test_labels)

    write_tensor(processed_dir / "train_images.bin", (train_image_shape[0], 28 * 28), train_images_flat)
    write_tensor(processed_dir / "train_labels.bin", (train_label_shape[0], 10), train_labels_oh)
    write_tensor(processed_dir / "test_images.bin", (test_image_shape[0], 28 * 28), test_images_flat)
    write_tensor(processed_dir / "test_labels.bin", (test_label_shape[0], 10), test_labels_oh)

    print("Summary:")
    print(f"  train images: {train_image_shape}, labels: {train_label_shape}")
    print(f"  test images: {test_image_shape}, labels: {test_label_shape}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target_root = repo_root / "datasets" / "mnist"
    prepare_dataset(target_root)


if __name__ == "__main__":
    main()
