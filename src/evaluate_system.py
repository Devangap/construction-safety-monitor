import os
from pathlib import Path
from typing import Dict, List, Any

import cv2
import pandas as pd
from ultralytics import YOLO
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from safety_logic import evaluate_scene
# from inference import parse_detections, CONF_THRESHOLD

MODEL_PATH = "models/best.pt"
EVAL_DIR = "eval_dataset"  # expected: eval_dataset/SAFE, eval_dataset/UNSAFE, eval_dataset/UNCERTAIN
OUTPUT_CSV = "system_evaluation_results.csv"

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
LABELS = ["SAFE", "UNSAFE", "UNCERTAIN"]
CONF_THRESHOLD = 0.5

CLASS_NAMES = {
    0: "person",
    1: "helmet",
    2: "vest",
}
def parse_detections(results):
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

def collect_images(eval_dir: str) -> List[Dict[str, str]]:
    items = []
    for label in LABELS:
        class_dir = Path(eval_dir) / label
        if not class_dir.exists():
            continue
        for p in class_dir.iterdir():
            if p.suffix.lower() in VALID_EXTS:
                items.append({
                    "image_path": str(p),
                    "true_label": label
                })
    return items


def predict_scene(model, image_path: str) -> Dict[str, Any]:
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

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

    return {
        "scene_status": scene_result["scene_status"],
        "total_persons": scene_result["total_persons"],
        "safe_workers": scene_result["safe_workers"],
        "unsafe_workers": scene_result["unsafe_workers"],
        "uncertain_workers": scene_result["uncertain_workers"],
        "worker_results": scene_result["worker_results"],
    }


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    samples = collect_images(EVAL_DIR)
    if not samples:
        raise ValueError(
            "No evaluation images found. Expected folders like "
            "eval_dataset/SAFE, eval_dataset/UNSAFE, eval_dataset/UNCERTAIN"
        )

    model = YOLO(MODEL_PATH)

    rows = []
    y_true = []
    y_pred = []

    for sample in samples:
        pred = predict_scene(model, sample["image_path"])

        true_label = sample["true_label"]
        pred_label = pred["scene_status"]

        rows.append({
            "image_name": Path(sample["image_path"]).name,
            "true_label": true_label,
            "predicted_label": pred_label,
            "correct": true_label == pred_label,
            "total_persons": pred["total_persons"],
            "safe_workers": pred["safe_workers"],
            "unsafe_workers": pred["unsafe_workers"],
            "uncertain_workers": pred["uncertain_workers"],
        })

        y_true.append(true_label)
        y_pred.append(pred_label)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\nSaved detailed evaluation to:", OUTPUT_CSV)
    print("\nAccuracy:", accuracy_score(y_true, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=LABELS, zero_division=0))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    cm_df = pd.DataFrame(cm, index=LABELS, columns=LABELS)
    print(cm_df)


if __name__ == "__main__":
    main()