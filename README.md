Construction Safety Monitor
 
AI-Powered PPE Compliance System
Overview
 
This project presents a computer vision-based construction safety monitoring system that analyzes site images to determine whether workers comply with essential Personal Protective Equipment (PPE) requirements.
Unlike traditional object detection systems, this solution goes beyond detection by performing worker-level reasoning. Each detected worker is individually evaluated for helmet and safety vest compliance, and the system produces a scene-level safety decision with explicit explanations.
The system is built using YOLOv8 + rule-based reasoning, combining deep learning with structured spatial logic to simulate real-world safety inspection.
 
Key Features
 
Worker detection
Helmet detection with head-region validation
Vest detection with torso-region validation
PPE-to-worker association
Worker-level violation detection
Uncertainty handling for small or distant workers
Scene-level safety classification
Annotated output images
CSV-based evaluation reports
Human-readable reasoning for every decision
 
 
System Architecture
Image -> YOLOv8 Detector -> Bounding Boxes -> PPE Association -> Worker Evaluation -> Scene Decision
1. Object Detection
Model: YOLOv8n
Classes:
 
person
helmet
vest
 
2. Worker-Level Reasoning
Each detected person is evaluated individually using spatial reasoning:
 
Helmet must belong to the head region
Vest must belong to the torso region
PPE items cannot be reused across workers
 
3. Decision Engine
Each worker is classified as: SAFE, UNSAFE, or UNCERTAIN
Scene-level decision:
 
UNSAFE — if any worker is unsafe
UNCERTAIN — if no unsafe workers but at least one is uncertain
SAFE — if all workers are compliant
 
 
Core Innovation — Safety Logic
A naive approach such as:
helmets >= persons -> SAFE
fails in real scenarios.
Example:
Persons: 2
Helmets: 2
Vests: 4
Naive Result: SAFE  (incorrect)
This system:
 
Matches PPE to each worker individually
Validates spatial position of each PPE item
Produces correct classification based on actual compliance
 
Decision Logic
pythonif has_helmet and has_vest:
    status = "SAFE"
elif small_person or low_confidence:
    status = "UNCERTAIN"
else:
    status = "UNSAFE"
```
 
---
 
## Safety Rules Definition
 
**Rule 1 — Worker Detection**
Each detected person is treated as a worker candidate.
 
**Rule 2 — Helmet Compliance**
A helmet is valid only if it is spatially associated with the head region of the worker.
 
**Rule 3 — Vest Compliance**
A vest is valid only if it is spatially associated with the torso region of the worker.
 
**Rule 4 — No PPE Reuse**
Each PPE item can only be assigned to one worker.
 
**Rule 5 — Worker Status**
 
- SAFE — helmet and vest present
- UNSAFE — missing one or both PPE items
- UNCERTAIN — small, distant, or visually unclear
 
**Rule 6 — Scene Status**
 
- UNSAFE — at least one unsafe worker
- UNCERTAIN — no unsafe workers, but at least one uncertain
- SAFE — all workers compliant
- NO_WORKER_DETECTED — no persons found
 
**Rule 7 — Confidence Filtering**
Only detections at or above 0.5 confidence are used.
 
---
 
## Handling Edge Cases
 
| Scenario | System Behavior |
|----------|----------------|
| Helmet in hand | UNSAFE |
| Small or distant worker | UNCERTAIN |
| Crowded scene | Conservative (bias toward UNSAFE) |
| Extra PPE in scene | Ignored unless matched to a worker |
 
---
 
## Example Inference Results
 
During inference, the system reports both scene-level and worker-level decisions, including PPE presence, compliance status, and explicit violation reasons. The detection counts shown below are produced by the YOLOv8 model.
 
### 1. Unsafe — Missing Helmet
```
Image: missing_helmet.jpg
Detected: 1 person, 0 helmets, 1 vest
Result: UNSAFE
Worker 1 -> Helmet: False, Vest: True, Status: UNSAFE, Reason: Missing helmet
```
 
### 2. Safe Scene
```
Image: safe.jpg
Detected: 1 person, 1 helmet, 1 vest
Result: SAFE
Worker 1 -> Helmet: True, Vest: True, Status: SAFE, Reason: Compliant
```
 
### 3. Multi-Worker with Uncertainty
```
Image: multiple_workers.jpeg
Detected: 4 persons, 4 helmets, 5 vests
Result: UNCERTAIN
Worker 3 -> Helmet: False, Vest: True, Status: UNCERTAIN, Reason: Helmet unclear
```
 
Shows uncertainty handling instead of forcing an incorrect SAFE classification.
 
### 4. Unsafe Multi-Worker Scene
```
Image: partially.jpeg
Detected: 7 persons, 5 helmets, 7 vests
Result: UNSAFE
Worker 2 -> Helmet: False, Vest: True, Status: UNSAFE, Reason: Missing helmet
Worker 4 -> Helmet: False, Vest: True, Status: UNSAFE, Reason: Missing helmet
```
 
Demonstrates worker-level violation detection in crowded scenes.
 
### 5. Mixed Unsafe and Uncertain
```
Image: unsafe.jpeg
Detected: 2 persons, 1 helmet, 0 vests
Result: UNSAFE
Worker 1 -> Helmet: True, Vest: False, Status: UNCERTAIN, Reason: Vest unclear
Worker 2 -> Helmet: False, Vest: False, Status: UNSAFE, Reason: Missing helmet, Missing vest
```
 
The system distinguishes between unclear and clearly unsafe workers.
 
### Visual Examples of Violations
 
Sample annotated outputs are provided in the `/outputs` folder, demonstrating:
 
- Missing helmet — classified as UNSAFE
- Missing vest — classified as UNSAFE
- Correct PPE usage — classified as SAFE
- Ambiguous cases — classified as UNCERTAIN
 
These examples illustrate how safety rules are applied in real-world scenarios.
 
---
 
## Model Performance
 
### Detection Metrics (YOLOv8) — from validation output
 
| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|-------|-----------|--------|--------|-----------|
| All | 0.853 | 0.749 | 0.812 | 0.521 |
| person | 0.897 | 0.766 | 0.826 | 0.545 |
| helmet | 0.892 | 0.757 | 0.842 | 0.524 |
| vest | 0.769 | 0.725 | 0.769 | 0.495 |
 
### System-Level Evaluation
 
| Metric | Score |
|--------|-------|
| Accuracy | 0.625 |
| UNSAFE Recall | 0.90 |
 
### Confusion Matrix
 
| Predicted / Actual | SAFE | UNSAFE | UNCERTAIN |
|-------------------|------|--------|-----------|
| **SAFE**          |  5   |   5    |     0     |
| **UNSAFE**        |  1   |   9    |     0     |
| **UNCERTAIN**     |  0   |   3    |     1     |
 
### Interpretation
 
The system is intentionally safety-first biased. It prioritizes detecting unsafe workers and accepts some false positives to avoid missing real violations. This aligns with real-world construction safety requirements.
 
The per-class breakdown shows that vest detection is the weakest of the three classes (mAP@50 of 0.769 vs 0.842 for helmet and 0.826 for person), which is consistent with the limitations noted below and supports the case for future improvements.
 
---
 
## Dataset Engineering
 
### Dataset Summary
 
| Split | Count |
|-------|-------|
| Train | 412 |
| Validation | 117 |
| Test | 60 |
| **Total** | **589** |
 
### Dataset Sources
 
- PPE Vest Helmet Dataset
- Helmet-Person-Vest Dataset
- Custom dataset — 141 images collected from Google Images and YouTube video frames
 
### Data Preparation
 
- Merged multiple datasets into a unified training set
- Standardized labels to: person, helmet, vest
- Converted annotations to YOLO format
- Remapped inconsistent labels into a common schema
- Created train / validation / test splits
 
### Why This Matters
 
The dataset was not used as-is. It was engineered to improve consistency, match real PPE scenarios, and support reliable reasoning.
 
### Data Diversity
 
The dataset includes a wide range of real-world conditions:
 
- Indoor and outdoor construction environments
- Daylight, shadow, and artificial lighting conditions
- Different worker poses, angles, and orientations
- Crowded scenes with multiple workers
- Occlusion cases (partially visible PPE)
- Both safe and unsafe PPE scenarios
 
This diversity improves the model's ability to generalize to real-world safety monitoring scenarios.
 
### Annotation Strategy
 
All annotations follow YOLO format: `(class_id, x_center, y_center, width, height)`
 
Class schema was standardized across datasets:
```
0 -> person
1 -> helmet
2 -> vest
```
 
- Segmentation annotations were converted to bounding boxes where required
- Label consistency was ensured by removing irrelevant or noisy classes, remapping inconsistent class indices, and validating annotation files after merging
 
This ensured compatibility with YOLOv8 training and reduced label noise.
 
### Dataset Bias and Trade-offs
 
The dataset is slightly biased toward visible PPE scenarios due to the nature of source datasets. Unsafe cases were intentionally increased through custom data collection. Vest detection remains more difficult due to occlusion, loose clothing variation, and color similarity with background.
 
The dataset prioritizes safety-critical detection (recall of unsafe cases) over perfect class balance, aligning with real-world construction safety requirements.
 
---
 
## Custom Dataset Collection and Annotation
 
To satisfy the requirement of using a custom dataset, an additional dataset of **141 images** was collected and integrated into the final training dataset.
 
### Data Collection
 
The custom dataset was collected from:
 
- Google Images (construction site scenarios)
- YouTube video frames (real-world construction environments)
 
The data collection process focused on capturing realistic and challenging scenarios, including:
 
- Workers without helmets
- Workers without safety vests
- Crowded construction scenes
- Occlusion and low-visibility conditions
 
This ensured the dataset includes both safe and unsafe PPE conditions, improving real-world applicability.
 
### Dataset Composition
 
Within the custom dataset:
 
| Class | Image Count |
|-------|-------------|
| Persons | 141 |
| Helmets | 104 |
| Vests | 81 |
 
Observations:
- Helmet instances are relatively higher than vest instances
- Vest annotations are lower, making vest detection more challenging
- This imbalance is reflected in the model evaluation, where vest detection shows the lowest performance
 
### Annotation Approach
 
The dataset was annotated using a semi-automated approach.
 
**Step 1 — Auto-Annotation**
Initial bounding boxes were generated using an automated annotation tool.
 
**Step 2 — Manual Verification and Correction**
All annotations were manually reviewed. Incorrect bounding boxes were corrected and missing annotations were added where necessary.
 
Special attention was given to:
- Ensuring helmets are correctly placed in the head region
- Ensuring vests are correctly aligned with the torso region
- Removing incorrect or noisy labels
 
### Annotation Quality Observations
 
Vest annotations were less frequent due to occlusion, loose clothing variations, and difficulty in visual detection. This introduces a class imbalance which contributes to lower detection performance for vests and increased difficulty in PPE compliance evaluation.
 
### Contribution to Final Dataset
 
The custom dataset significantly improves the overall system by:
 
- Introducing unsafe scenarios not well represented in public datasets
- Increasing dataset diversity
- Improving model robustness under real-world conditions
- Supporting better evaluation of safety rule logic
 
The custom dataset was intentionally designed to include realistic unsafe scenarios and challenging edge cases, rather than relying solely on ideal or pre-annotated benchmark data.
 
---
 
## Limitations
 
- Small or distant workers — unclear PPE visibility
- Occlusion — incorrect PPE region estimation
- Heuristic body regions — not pose-aware
- Detection dependency — errors propagate from the model
- Crowded scenes — ambiguous PPE assignment
- Vest detection is the weakest class, as confirmed by per-class validation metrics
 
---
 
## Internal Uncertainty
 
UNCERTAIN is used as an internal reasoning state to represent ambiguity. In real deployment, UNCERTAIN can be treated as UNSAFE for safety-critical environments.
 
---
```
## Project Structure
 
```
construction-safety-monitor/
├── models/
│   └── best.pt (YOLOv8 trained model)
├── sample_inputs/
├── outputs/
├── eval_dataset/
│   ├── SAFE/
│   ├── UNSAFE/
│   └── UNCERTAIN/
├── src/
│   ├── predict.py (Inference engine)
│   ├── evaluate_system.py (Evaluation script)
│   ├── safety_logic.py (Core reasoning logic)
│   └── __init__.py
├── requirements.txt
└── README.md
```
## How to Run
 
### 1. Prerequisites
 
Ensure you have the following installed:
 
- Python 3.8+
- pip
 
Install dependencies:
 
```bash
pip install -r requirements.txt
```
 
### 2. Run Inference
 
```bash
python src/predict.py
```
 
**Outputs:**
 
- Annotated images (saved in `/outputs/`)
- Worker-level reasoning
- Scene-level classification (SAFE / UNSAFE / UNCERTAIN)
 
### 3. Run Evaluation
 
```bash
python src/evaluate_system.py
```
 
**Expected dataset structure:**
 
```
eval_dataset/
├── SAFE/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── UNSAFE/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── UNCERTAIN/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```
 
**Outputs will include:**
 
- CSV report
- Accuracy score
- Confusion matrix
- Classification report
 
### 4. Run in Google Colab (Optional)
 
Upload project files to Colab and install dependencies:
 
```bash
pip install ultralytics opencv-python pandas scikit-learn
```
 
- Update dataset paths if needed (`/content/...`)
- Run the same commands above
 
---
 
## Tech Stack
 
- Python
- YOLOv8 (Ultralytics)
- OpenCV
- NumPy
- Pandas
- scikit-learn
- Google Colab
 
---
 
## What Makes This Project Stand Out
 
- Goes beyond detection — reasoning-based system design
- Worker-level safety evaluation
- No PPE reuse constraint
- Explicit explainability at the worker level
- Safety-first decision design
- Full system evaluation beyond raw detection metrics
 
---
 
## Future Improvements
 
- Pose estimation for better PPE localization
- Learned PPE-to-worker matching instead of heuristics
- Larger dataset with more hard cases
- Real-time video processing
- Deployment via FastAPI or Streamlit
 
## Extended Violation Categories (Future Work)
 
While the current system focuses on PPE presence and correct spatial association,  
additional safety violations such as:
 
- Helmet not fastened  
- Vest worn open or improperly  
- Unsafe worker posture (e.g., working at height without fall protection)  
 
are not currently detected.
 
These require advanced techniques such as:
- Pose estimation (e.g., keypoint detection)
- Fine-grained PPE condition classification
- Context-aware scene understanding
 
Future versions of the system can integrate these capabilities to improve real-world safety monitoring.
 
---
 
## Final Pipeline
```
Data -> Detection -> Reasoning -> Decision -> Evaluation
 
 