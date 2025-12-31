# 🎾 Tennis Hit & Bounce Detection

This project implements **frame-level detection of tennis ball events** — **air**, **hit**, and **bounce** — using both **supervised machine learning** and an **unsupervised physics-based approach**.

Ball trajectories are extracted from video as `(x, y)` coordinates per frame. Temporal context is modeled using **sliding windows of kinematic features** to accurately detect short, rare events such as racket hits and ground bounces.

---

## 📂 Project Structure

tennis-hit-bounce-detection/
│
├── dataset.py # Dataset construction & feature extraction
├── main.py # Supervised & unsupervised inference pipelines
├── main.ipynb # Training, evaluation & analysis notebook
├── trained_supervised.joblib # Trained supervised model (XGBoost)
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── pycache/ # Python cache files


---

## 🧠 Problem Overview

Each tennis point is represented as a time series of ball positions `(x, y)` extracted from video frames.

Each frame is labeled as:
- **air** – normal ball flight  
- **hit** – racket contact  
- **bounce** – ground contact  

### Key Challenges
- Extreme **class imbalance** (most frames are `air`)
- Events (`hit`, `bounce`) are **short and rare**
- Detection requires **temporal context**, not single-frame reasoning

---

## 📊 Feature Engineering

For each visible frame, kinematic features are computed:
(x, y, vx, vy, ax, ay)


Where:
- `vx, vy` = velocity
- `ax, ay` = acceleration

### Sliding Windows
To capture motion context, sliding windows are applied:
- Window sizes tested: **1, 3, 5, 7**
- Best performance achieved with **window size = 5**

Each window is flattened into a single feature vector.

---

## 🧩 Dataset Module (`dataset.py`)

### `TennisKinematicsWindowDataset`

Responsible for:
- Loading JSON trajectory files
- Filtering non-visible frames
- Computing kinematic features
- Creating sliding-window samples

### Supervised Learning Approach

The supervised pipeline treats hit and bounce detection as a multi-class classification problem at the frame level.

Models Evaluated: 
- Random forest
- XGBoost (best performing)

Training Strategy:

Sliding-window classification over kinematic features
Cost-sensitive learning to handle severe class imbalance
Optimization metric: macro F1-score on hit & bounce classes only

⚖️ Class Imbalance Handling

The dataset is heavily dominated by the air class. To address this, class weights are applied during training to penalize misclassification of rare events more strongly.

Best Configuration (XGBoost)

- Window size: 5
- Class weights: {0: 1.0, 1: 7.0, 2: 12.0}

Supervised Inference (main.py)
supervized_hit_bounce_detection(ball_data_i)

This function applies the trained supervised model to a single ball trajectory file.

- Loads a JSON trajectory file
- Extracts sliding-window features
- Applies the trained classifier
- Assigns predictions only to center frames
- Non-visible frames are always labeled "air"

Each frame is augmented with:

"pred_action": "air" | "hit" | "bounce"

🧪 Unsupervised Physics-Based Approach

A fully unsupervised baseline based on physical motion cues, requiring no labeled data.

Signals Used:
- Velocity and acceleration
- Speed and speed change
- Direction (angle) change between velocity vectors
- Event Detection Logic
- Bounce detection
- Ball near the ground
- Local vertical extremum
- Sign change in vertical velocity
- Strong vertical acceleration impulse

Hit detection;

- Sudden direction change
- Large speed or acceleration variation
- Explicitly excludes near-ground frames
- Additional Constraints
- Adaptive percentile-based thresholds
- Cooldown period to avoid duplicate detections

Unsupervised Performance Summary:

- Excellent detection of air frames
- Moderate recall for hit
- Very low recall for bounce
- While interpretable and robust, this method lacks the discriminative power needed for reliable rare-event detection.

Experiments & Analysis (main.ipynb)
The notebook contains:

- Dataset construction and visualization
- Baseline Random Forest training
- Class-weight tuning
- Sliding-window size optimization
- XGBoost grid search
- Confusion matrices and detailed metrics
- Supervised vs unsupervised comparison
- Final model training and serialization