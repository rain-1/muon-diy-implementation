from array import array
from pathlib import Path
from typing import Iterable, Sequence

from datasets import load_dataset
from PIL import Image


REPO_ID = "uoft-cs/cifar100"


def normalize_images(images: Iterable[Image.Image]) -> array:
    data = array("f")
    for img in images:
        if img.mode != "RGB":
            img = img.convert("RGB")
        if img.size != (32, 32):
            raise ValueError(f"Unexpected image size {img.size}; expected 32x32")
        for pixel in img.getdata():
            r, g, b = pixel
            data.extend((r / 255.0, g / 255.0, b / 255.0))
    return data


def to_one_hot(labels: Sequence[int], num_classes: int = 100) -> array:
    result = array("f", [0.0] * (len(labels) * num_classes))
    for idx, label in enumerate(labels):
        if label < 0 or label >= num_classes:
            raise ValueError(f"Label out of range: {label}")
        result[idx * num_classes + label] = 1.0
    return result


def write_tensor(path: Path, shape: Sequence[int], data: array) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = array("i", [len(shape), *shape])
    with open(path, "wb") as f:
        header.tofile(f)
        data.tofile(f)
    print(f"Wrote tensor {tuple(shape)} -> {path}")


def prepare_dataset(root: Path) -> None:
    print(f"Loading CIFAR-100 from {REPO_ID} ...")
    dataset = load_dataset(REPO_ID)

    train = dataset["train"]
    test = dataset["test"]

    train_images = normalize_images(train["image"])
    test_images = normalize_images(test["image"])

    train_labels = [int(label) for label in train["fine_label"]]
    test_labels = [int(label) for label in test["fine_label"]]

    train_labels_oh = to_one_hot(train_labels)
    test_labels_oh = to_one_hot(test_labels)

    processed = root / "processed"
    write_tensor(processed / "train_images.bin", (len(train_labels), 32 * 32 * 3), train_images)
    write_tensor(processed / "train_labels.bin", (len(train_labels), 100), train_labels_oh)
    write_tensor(processed / "test_images.bin", (len(test_labels), 32 * 32 * 3), test_images)
    write_tensor(processed / "test_labels.bin", (len(test_labels), 100), test_labels_oh)

    print("Summary:")
    print(f"  train images: {(len(train_labels), 3, 32, 32)}, labels: {(len(train_labels),)}")
    print(f"  test images: {(len(test_labels), 3, 32, 32)}, labels: {(len(test_labels),)}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    target_root = repo_root / "datasets" / "cifar100"
    prepare_dataset(target_root)


if __name__ == "__main__":
    main()
