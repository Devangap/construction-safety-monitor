import os
from pathlib import Path
from typing import List, Dict, Any

import cv2
from ultralytics import YOLO

from safety_logic import evaluate_scene


MODEL_PATH = "models/best.pt"
INPUT_DIR = "sample_inputs"
OUTPUT_DIR = "outputs"
CONF_THRESHOLD = 0.5

CLASS_NAMES = {
    0: "person",
    1: "helmet",
    2: "vest",
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_detections(results) -> Dict[str, List[Dict[str, Any]]]:
    parsed = {
        "persons": [],
        "helmets": [],
        "vests": [],
    }

    boxes = results.boxes
    if boxes is None:
        return parsed

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    class_ids = boxes.cls.cpu().numpy().astype(int)

    for box, conf, class_id in zip(xyxy, confs, class_ids):
        if conf < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(float, box)
        label = CLASS_NAMES.get(class_id, f"class_{class_id}")

        det = {
            "box": (x1, y1, x2, y2),
            "confidence": float(conf),
            "class_id": class_id,
            "label": label,
        }

        if label == "person":
            parsed["persons"].append(det)
        elif label == "helmet":
            parsed["helmets"].append(det)
        elif label == "vest":
            parsed["vests"].append(det)

    return parsed


def draw_box(image, box, text, color, thickness=2):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    label_y = max(y1 - 10, 20)
    cv2.putText(
        image,
        text,
        (x1, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def annotate_image(image, detections, scene_result):
    annotated = image.copy()

    for person in detections["persons"]:
        draw_box(
            annotated,
            person["box"],
            f"person {person['confidence']:.2f}",
            (255, 255, 0),
            2,
        )

    for helmet in detections["helmets"]:
        draw_box(
            annotated,
            helmet["box"],
            f"helmet {helmet['confidence']:.2f}",
            (0, 255, 0),
            2,
        )

    for vest in detections["vests"]:
        draw_box(
            annotated,
            vest["box"],
            f"vest {vest['confidence']:.2f}",
            (255, 0, 0),
            2,
        )

    for worker in scene_result["worker_results"]:
        if worker["status"] == "SAFE":
            color = (0, 255, 0)
        elif worker["status"] == "UNCERTAIN":
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)

        label = f"Worker {worker['person_id']} - {worker['status']}"
        if worker["reason"] != "Compliant":
            label += f": {worker['reason']}"

        draw_box(annotated, worker["person_box"], label, color, 3)

    if scene_result["scene_status"] == "SAFE":
        scene_color = (0, 255, 0)
    elif scene_result["scene_status"] == "UNCERTAIN":
        scene_color = (0, 255, 255)
    elif scene_result["scene_status"] == "UNSAFE":
        scene_color = (0, 0, 255)
    else:
        scene_color = (255, 255, 255)

    scene_text = f"FINAL STATUS: {scene_result['scene_status']}"
    cv2.putText(
        annotated,
        scene_text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        scene_color,
        3,
        cv2.LINE_AA,
    )

    return annotated


def print_summary(image_name, detections, scene_result):
    print("=" * 70)
    print(f"Image: {image_name}")
    print(f"Persons detected: {len(detections['persons'])}")
    print(f"Helmets detected: {len(detections['helmets'])}")
    print(f"Vests detected: {len(detections['vests'])}")
    print(f"Safe workers: {scene_result['safe_workers']}")
    print(f"Unsafe workers: {scene_result['unsafe_workers']}")
    print(f"Uncertain workers: {scene_result['uncertain_workers']}")
    print(f"FINAL STATUS: {scene_result['scene_status']}")

    for worker in scene_result["worker_results"]:
        print(
            f"Worker {worker['person_id']} -> "
            f"Helmet: {worker['has_helmet']}, "
            f"Vest: {worker['has_vest']}, "
            f"Status: {worker['status']}, "
            f"Reason: {worker['reason']}"
        )


def run_on_image(model, image_path: str):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return

    results = model.predict(
        source=image_path,
        conf=CONF_THRESHOLD,
        verbose=False,
    )[0]

    detections = parse_detections(results)

    scene_result = evaluate_scene(
        persons=detections["persons"],
        helmets=detections["helmets"],
        vests=detections["vests"],
        image_shape=image.shape[:2],
    )

    annotated = annotate_image(image, detections, scene_result)

    output_path = os.path.join(OUTPUT_DIR, Path(image_path).name)
    cv2.imwrite(output_path, annotated)

    print_summary(Path(image_path).name, detections, scene_result)
    print(f"Saved annotated output to: {output_path}")


def main():
    ensure_dir(OUTPUT_DIR)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not os.path.exists(INPUT_DIR):
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    model = YOLO(MODEL_PATH)

    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}
    image_paths = [
        str(p) for p in Path(INPUT_DIR).iterdir()
        if p.suffix.lower() in image_extensions
    ]

    if not image_paths:
        print("No input images found in sample_inputs/")
        return

    for image_path in sorted(image_paths):
        run_on_image(model, image_path)


if __name__ == "__main__":
    main()