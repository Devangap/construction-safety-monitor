import random
import shutil
from pathlib import Path

# ===== PATHS =====
dataset1 = Path("/Users/dev/Downloads/PPE vest helmet.yolov8-2/train")
dataset2 = Path("/Users/dev/Downloads/helmet-person-vest.yolov8-2/train")
dataset3 = Path("/Users/dev/Downloads/PPE object detection dev.yolov8/train")  # your custom Roboflow export

output_dataset = Path("/Users/dev/Desktop/construction-safety-monitor/final_dataset_all")

# ===== SETTINGS =====
image_exts = [".jpg", ".jpeg", ".png"]
train_ratio = 0.7
valid_ratio = 0.2
test_ratio = 0.1

# Final class order:
# 0 = person
# 1 = helmet
# 2 = vest
remap = {0: 1, 1: 0, 2: 2}
final_names = ["person", "helmet", "vest"]

random.seed(42)


def make_dirs():
    for split in ["train", "valid", "test"]:
        (output_dataset / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dataset / split / "labels").mkdir(parents=True, exist_ok=True)


def get_image_files(images_dir):
    files = []
    for ext in image_exts:
        files.extend(images_dir.glob(f"*{ext}"))
        files.extend(images_dir.glob(f"*{ext.upper()}"))
    return files


def polygon_to_bbox(coords):
    xs = coords[0::2]
    ys = coords[1::2]

    x_min = max(0.0, min(xs))
    y_min = max(0.0, min(ys))
    x_max = min(1.0, max(xs))
    y_max = min(1.0, max(ys))

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    width = x_max - x_min
    height = y_max - y_min

    return x_center, y_center, width, height


def convert_label_file_to_detection(src_label_path, dst_label_path):
    if not src_label_path.exists():
        dst_label_path.write_text("")
        return

    new_lines = []

    with open(src_label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            try:
                old_class = int(float(parts[0]))
            except ValueError:
                continue

            if old_class not in remap:
                continue

            new_class = remap[old_class]

            # Detection format: class x_center y_center width height
            if len(parts) == 5:
                try:
                    x_center, y_center, width, height = map(float, parts[1:5])
                except ValueError:
                    continue

            # Segmentation format: class x1 y1 x2 y2 ...
            elif len(parts) > 5 and (len(parts) - 1) % 2 == 0:
                try:
                    coords = list(map(float, parts[1:]))
                except ValueError:
                    continue
                x_center, y_center, width, height = polygon_to_bbox(coords)

            else:
                continue

            # skip invalid boxes
            if width <= 0 or height <= 0:
                continue

            new_line = f"{new_class} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
            new_lines.append(new_line)

    with open(dst_label_path, "w") as f:
        if new_lines:
            f.write("\n".join(new_lines) + "\n")
        else:
            f.write("")


def collect_samples(dataset_train_path, prefix):
    images_dir = dataset_train_path / "images"
    labels_dir = dataset_train_path / "labels"

    samples = []
    for img_path in get_image_files(images_dir):
        label_path = labels_dir / f"{img_path.stem}.txt"
        samples.append({
            "img": img_path,
            "label": label_path,
            "new_stem": f"{prefix}_{img_path.stem}",
            "ext": img_path.suffix
        })
    return samples


def split_samples(samples):
    random.shuffle(samples)
    n = len(samples)

    n_train = int(n * train_ratio)
    n_valid = int(n * valid_ratio)

    train_samples = samples[:n_train]
    valid_samples = samples[n_train:n_train + n_valid]
    test_samples = samples[n_train + n_valid:]

    return train_samples, valid_samples, test_samples


def copy_split(samples, split_name):
    for sample in samples:
        dst_img = output_dataset / split_name / "images" / f"{sample['new_stem']}{sample['ext']}"
        dst_lbl = output_dataset / split_name / "labels" / f"{sample['new_stem']}.txt"

        shutil.copy2(sample["img"], dst_img)
        convert_label_file_to_detection(sample["label"], dst_lbl)


def create_data_yaml():
    yaml_text = f"""train: train/images
val: valid/images
test: test/images

nc: 3
names: {final_names}
"""
    with open(output_dataset / "data.yaml", "w") as f:
        f.write(yaml_text)


def main():
    make_dirs()

    samples1 = collect_samples(dataset1, "ds1")
    samples2 = collect_samples(dataset2, "ds2")
    samples3 = collect_samples(dataset3, "ds3")

    print(f"Dataset 1 images: {len(samples1)}")
    print(f"Dataset 2 images: {len(samples2)}")
    print(f"Dataset 3 images: {len(samples3)}")

    all_samples = samples1 + samples2 + samples3
    print(f"Total images found: {len(all_samples)}")

    train_samples, valid_samples, test_samples = split_samples(all_samples)

    print(f"Train: {len(train_samples)}")
    print(f"Valid: {len(valid_samples)}")
    print(f"Test:  {len(test_samples)}")

    copy_split(train_samples, "train")
    copy_split(valid_samples, "valid")
    copy_split(test_samples, "test")

    create_data_yaml()

    print(f"\nDone. Final detection dataset created at:\n{output_dataset}")


if __name__ == "__main__":
    main()