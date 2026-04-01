from typing import List, Dict, Tuple, Any, Optional, Set
import math

Box = Tuple[float, float, float, float]  # (x1, y1, x2, y2)


def get_box_center(box: Box) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def point_in_box(point: Tuple[float, float], box: Box) -> bool:
    px, py = point
    x1, y1, x2, y2 = box
    return x1 <= px <= x2 and y1 <= py <= y2


def box_area(box: Box) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def intersection_area(box_a: Box, box_b: Box) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    return (inter_x2 - inter_x1) * (inter_y2 - inter_y1)


def overlap_ratio(inner_box: Box, outer_box: Box) -> float:
    """
    Returns how much of inner_box lies inside outer_box.
    Useful for checking whether helmet/vest is meaningfully inside a target region.
    """
    inner_area = box_area(inner_box)
    if inner_area == 0:
        return 0.0
    return intersection_area(inner_box, outer_box) / inner_area


def distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def get_head_region(person_box: Box) -> Box:
    """
    Approximate head region from the upper central part of a person's bounding box.
    """
    x1, y1, x2, y2 = person_box
    w = x2 - x1
    h = y2 - y1

    head_x1 = x1 + 0.2 * w
    head_x2 = x2 - 0.2 * w
    head_y1 = y1
    head_y2 = y1 + 0.25 * h

    return (head_x1, head_y1, head_x2, head_y2)


def get_torso_region(person_box: Box) -> Box:
    """
    Approximate vest region from upper-middle body.
    """
    x1, y1, x2, y2 = person_box
    w = x2 - x1
    h = y2 - y1

    torso_x1 = x1 + 0.15 * w
    torso_x2 = x2 - 0.15 * w
    torso_y1 = y1 + 0.22 * h
    torso_y2 = y1 + 0.65 * h

    return (torso_x1, torso_y1, torso_x2, torso_y2)


def is_small_person(person_box: Box, image_shape: Optional[Tuple[int, int]] = None) -> bool:
    """
    Marks a worker as small/distant only if the box is genuinely tiny
    relative to the image.
    """
    if image_shape is None:
        return False

    img_h, img_w = image_shape
    if img_h <= 0 or img_w <= 0:
        return False

    x1, y1, x2, y2 = person_box
    person_w = max(0.0, x2 - x1)
    person_h = max(0.0, y2 - y1)
    person_area = person_w * person_h
    img_area = img_h * img_w

    rel_area = person_area / img_area
    rel_height = person_h / img_h
    rel_width = person_w / img_w

    return (
        rel_area < 0.003 or
        rel_height < 0.12 or
        rel_width < 0.04
    )


def is_helmet_for_person(helmet_box: Box, person_box: Box) -> bool:
    head_region = get_head_region(person_box)
    center = get_box_center(helmet_box)

    center_match = point_in_box(center, head_region)
    overlap_match = overlap_ratio(helmet_box, head_region) >= 0.2

    return center_match or overlap_match


def is_vest_for_person(vest_box: Box, person_box: Box) -> bool:
    torso_region = get_torso_region(person_box)
    center = get_box_center(vest_box)

    center_match = point_in_box(center, torso_region)
    overlap_match = overlap_ratio(vest_box, torso_region) >= 0.2

    return center_match or overlap_match


def find_best_helmet(
    person_box: Box,
    helmets: List[Dict[str, Any]],
    used_helmet_ids: Set[int],
) -> Optional[Dict[str, Any]]:
    head_region = get_head_region(person_box)
    head_center = get_box_center(head_region)

    best_helmet = None
    min_dist = float("inf")

    for idx, helmet in enumerate(helmets):
        if idx in used_helmet_ids:
            continue

        helmet_box = helmet["box"]
        helmet_center = get_box_center(helmet_box)

        center_match = point_in_box(helmet_center, head_region)
        overlap_match = overlap_ratio(helmet_box, head_region) >= 0.2

        if not (center_match or overlap_match):
            continue

        dist = distance(head_center, helmet_center)
        if dist < min_dist:
            min_dist = dist
            best_helmet = {"index": idx, **helmet}

    return best_helmet


def find_best_vest(
    person_box: Box,
    vests: List[Dict[str, Any]],
    used_vest_ids: Set[int],
) -> Optional[Dict[str, Any]]:
    torso_region = get_torso_region(person_box)
    torso_center = get_box_center(torso_region)

    best_vest = None
    min_dist = float("inf")

    for idx, vest in enumerate(vests):
        if idx in used_vest_ids:
            continue

        vest_box = vest["box"]
        vest_center = get_box_center(vest_box)

        center_match = point_in_box(vest_center, torso_region)
        overlap_match = overlap_ratio(vest_box, torso_region) >= 0.2

        if not (center_match or overlap_match):
            continue

        dist = distance(torso_center, vest_center)
        if dist < min_dist:
            min_dist = dist
            best_vest = {"index": idx, **vest}

    return best_vest


def evaluate_worker(
    person_box: Box,
    helmets: List[Dict[str, Any]],
    vests: List[Dict[str, Any]],
    person_id: int,
    used_helmet_ids: Set[int],
    used_vest_ids: Set[int],
    image_shape: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    matched_helmet = find_best_helmet(person_box, helmets, used_helmet_ids)
    matched_vest = find_best_vest(person_box, vests, used_vest_ids)

    if matched_helmet is not None:
        used_helmet_ids.add(matched_helmet["index"])
    if matched_vest is not None:
        used_vest_ids.add(matched_vest["index"])

    has_helmet = matched_helmet is not None
    has_vest = matched_vest is not None

    small_person = is_small_person(person_box, image_shape)
    LOW_CONF = 0.6

    helmet_low_conf = matched_helmet is not None and matched_helmet["confidence"] < LOW_CONF
    vest_low_conf = matched_vest is not None and matched_vest["confidence"] < LOW_CONF

    reasons = []

    if has_helmet and has_vest:
        status = "SAFE"
        reason = "Compliant"

    elif small_person or helmet_low_conf or vest_low_conf:
        status = "UNCERTAIN"
        if not has_helmet:
            reasons.append("Helmet unclear")
        if not has_vest:
            reasons.append("Vest unclear")
        reason = ", ".join(reasons)

    else:
        status = "UNSAFE"
        if not has_helmet:
            reasons.append("Missing helmet")
        if not has_vest:
            reasons.append("Missing vest")
        reason = ", ".join(reasons)

    return {
        "person_id": person_id,
        "person_box": person_box,
        "has_helmet": has_helmet,
        "has_vest": has_vest,
        "status": status,
        "reason": reason,
    }


def evaluate_scene(
    persons: List[Dict[str, Any]],
    helmets: List[Dict[str, Any]],
    vests: List[Dict[str, Any]],
    image_shape: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    worker_results = []
    used_helmet_ids: Set[int] = set()
    used_vest_ids: Set[int] = set()

    for idx, person in enumerate(persons, start=1):
        worker_result = evaluate_worker(
            person_box=person["box"],
            helmets=helmets,
            vests=vests,
            person_id=idx,
            used_helmet_ids=used_helmet_ids,
            used_vest_ids=used_vest_ids,
            image_shape=image_shape,
        )
        worker_results.append(worker_result)

    total_persons = len(persons)
    safe_workers = sum(1 for w in worker_results if w["status"] == "SAFE")
    unsafe_workers = sum(1 for w in worker_results if w["status"] == "UNSAFE")
    uncertain_workers = sum(1 for w in worker_results if w["status"] == "UNCERTAIN")

    if total_persons == 0:
        scene_status = "NO_WORKER_DETECTED"
    elif unsafe_workers > 0:
        scene_status = "UNSAFE"
    elif uncertain_workers > 0:
        scene_status = "UNCERTAIN"
    else:
        scene_status = "SAFE"

    return {
        "scene_status": scene_status,
        "total_persons": total_persons,
        "safe_workers": safe_workers,
        "unsafe_workers": unsafe_workers,
        "uncertain_workers": uncertain_workers,
        "worker_results": worker_results,
    }