import gzip
import struct
import urllib.request
from array import array
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

MNIST_URLS = {
    "train_images": "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
    "test_images": "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels": "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
}


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"Skipping download, found existing {dest}")
        return
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)


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


def prepare_dataset(root: Path) -> None:
    download_dir = root / "raw"
    processed_dir = root / "processed"
    for key, url in MNIST_URLS.items():
        download_file(url, download_dir / f"{key}.gz")

    train_pixels, train_image_shape = read_images(download_dir / "train_images.gz")
    test_pixels, test_image_shape = read_images(download_dir / "test_images.gz")
    train_labels, train_label_shape = read_labels(download_dir / "train_labels.gz")
    test_labels, test_label_shape = read_labels(download_dir / "test_labels.gz")

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
