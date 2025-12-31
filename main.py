import os
import json
import joblib
import numpy as np
from dataset import TennisKinematicsWindowDataset

INV_LABEL_MAP = {0: "air", 1: "hit", 2: "bounce"}


def supervized_hit_bounce_detection(ball_data_i):
    """
    Apply the trained supervised model to one ball_data JSON file
    and add a 'pred_action' field to each frame.

    Non-visible frames are always labeled as 'air'.
    """

    folder_path = r"C:\Users\adamh\Desktop\Projects\tennis-hit-bounce-detection\per_point_v2"
    model_path = r"C:\Users\adamh\Desktop\Projects\tennis-hit-bounce-detection\trained_supervised.joblib"


    window_size = 5
    half = window_size // 2

    json_path = os.path.join(folder_path, ball_data_i)

    # ---- Load JSON ----
    with open(json_path, "r") as f:
        data = json.load(f)

    # ---- Initialize ALL frames as "air" ----
    for fr in data.keys():
        data[str(fr)]["pred_action"] = "air"

    # ---- Collect visible frames ----
    frames = sorted(map(int, data.keys()))
    visible_frames = []

    for fr in frames:
        e = data[str(fr)]
        if e["visible"] and e["x"] is not None and e["y"] is not None:
            visible_frames.append(fr)

    # Not enough frames → return early
    if len(visible_frames) < window_size:
        return data

    # ---- Build features for this file ----
    dataset = TennisKinematicsWindowDataset("", window_size=window_size)
    X, _ = dataset.load_file(json_path)

    # ---- Load model ----
    clf = joblib.load(model_path)

    # ---- Predict ----
    preds = clf.predict(X)

    # ---- Assign predictions to center frames only ----
    pred_idx = 0
    for i in range(half, len(visible_frames) - half):
        fr = visible_frames[i]
        data[str(fr)]["pred_action"] = INV_LABEL_MAP[preds[pred_idx]]
        pred_idx += 1

    return data




def unsupervised_hit_bounce_detection(ball_data_i):
    """
    Apply an unsupervised physics-based method to one ball_data JSON file
    and add a 'pred_action' field to each frame.

    Non-visible frames are always labeled as 'air'.
    """

    folder_path = r"C:\Users\adamh\Desktop\Projects\tennis-hit-bounce-detection\per_point_v2"
    json_path = os.path.join(folder_path, ball_data_i)

    with open(json_path, "r") as f:
        data = json.load(f)

    # Initialize all frames as air
    for fr in data.keys():
        data[str(fr)]["pred_action"] = "air"

    frames = sorted(map(int, data.keys()))

    xs, ys, valid_frames = [], [], []

    for fr in frames:
        e = data[str(fr)]
        if e["visible"] and e["x"] is not None and e["y"] is not None:
            xs.append(float(e["x"]))
            ys.append(float(e["y"]))
            valid_frames.append(fr)

    if len(xs) < 7:
        return data

    xs = np.array(xs)
    ys = np.array(ys)

    vx = np.gradient(xs)
    vy = np.gradient(ys)
    ax = np.gradient(vx)
    ay = np.gradient(vy)

    speed = np.sqrt(vx**2 + vy**2)
    speed_change = np.abs(np.gradient(speed))
    acc_mag = np.sqrt(ax**2 + ay**2)

    # Infer ground direction automatically
    if np.median(ys) > np.mean(ys):
        near_ground = ys >= np.percentile(ys, 85)
        ground_extrema = (ys[1:-1] >= ys[:-2]) & (ys[1:-1] >= ys[2:])
    else:
        near_ground = ys <= np.percentile(ys, 15)
        ground_extrema = (ys[1:-1] <= ys[:-2]) & (ys[1:-1] <= ys[2:])

    ay_thresh = np.percentile(np.abs(ay), 90)
    speed_thresh = np.percentile(speed_change, 90)
    acc_thresh = np.percentile(acc_mag, 90)

    v = np.stack([vx, vy], axis=1)
    v_norm = np.linalg.norm(v, axis=1) + 1e-8
    cos_angle = np.sum(v[1:] * v[:-1], axis=1) / (v_norm[1:] * v_norm[:-1])
    angle_change = np.zeros(len(xs))
    angle_change[1:] = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    angle_thresh = np.percentile(angle_change, 85)

    labels = ["air"] * len(xs)
    cooldown = 4
    last_event = -cooldown

    for i in range(1, len(xs) - 1):
        if i - last_event < cooldown:
            continue

        # Bounce
        if (
            near_ground[i]
            and ground_extrema[i - 1]
            and abs(ay[i]) > ay_thresh
            and vy[i - 1] * vy[i] < 0
        ):
            labels[i] = "bounce"
            last_event = i
            continue

        # Hit
        if (
            not near_ground[i]
            and angle_change[i] > angle_thresh
            and (speed_change[i] > speed_thresh or acc_mag[i] > acc_thresh)
        ):
            labels[i] = "hit"
            last_event = i

    for fr, pred in zip(valid_frames, labels):
        data[str(fr)]["pred_action"] = pred

    return data
