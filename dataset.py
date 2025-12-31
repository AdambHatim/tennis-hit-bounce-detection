import os
import json
import numpy as np

LABEL_MAP = {"air": 0, "hit": 1, "bounce": 2}


class TennisKinematicsWindowDataset:
    def __init__(self, data_dir, window_size=5):
        assert window_size % 2 == 1, "window_size must be odd"
        self.data_dir = data_dir
        self.window_size = window_size
        self.half = window_size // 2

        if data_dir != "":
            self.files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".json")
        ])

    def _compute_kinematics(self, x, y):
        vx = np.gradient(x)
        vy = np.gradient(y)
        ax = np.gradient(vx)
        ay = np.gradient(vy)
        return vx, vy, ax, ay

    def load_all(self):
        X_all, y_all = [], []

        for path in self.files:
            with open(path, "r") as f:
                data = json.load(f)

            frames = sorted(map(int, data.keys()))

            xs, ys, labels = [], [], []
            for fr in frames:
                e = data[str(fr)]
                if not e["visible"]:
                    continue
                if e["x"] is None or e["y"] is None:
                    continue
                xs.append(float(e["x"]))
                ys.append(float(e["y"]))
                labels.append(e["action"])

            xs = np.array(xs)
            ys = np.array(ys)
            labels = np.array(labels)

            if len(xs) < self.window_size:
                continue

            # --- kinematics ---
            vx, vy, ax, ay = self._compute_kinematics(xs, ys)

            feats = np.stack([xs, ys, vx, vy, ax, ay], axis=1)  # (T, 6)

            # --- sliding windows ---
            for i in range(self.half, len(feats) - self.half):
                window = feats[i - self.half:i + self.half + 1].reshape(-1)
                X_all.append(window)
                y_all.append(LABEL_MAP[labels[i]])

        return np.array(X_all), np.array(y_all)
    
    def load_file(self, file_path):
            """
            Load a single JSON file and build sliding-window features.

            Parameters
            ----------
            file_path : str
                Path to a single ball_data_*.json file

            Returns
            -------
            X : np.ndarray, shape (N, window_size * 6)
                Feature matrix
            y : np.ndarray, shape (N,)
                Labels (air/hit/bounce encoded)
            """

            X, y = [], []

            with open(file_path, "r") as f:
                data = json.load(f)

            frames = sorted(map(int, data.keys()))

            xs, ys, labels = [], [], []
            for fr in frames:
                e = data[str(fr)]
                if not e["visible"]:
                    continue
                if e["x"] is None or e["y"] is None:
                    continue
                xs.append(float(e["x"]))
                ys.append(float(e["y"]))
                labels.append(e["action"])

            xs = np.array(xs)
            ys = np.array(ys)
            labels = np.array(labels)

            if len(xs) < self.window_size:
                return np.empty((0, self.window_size * 6)), np.empty((0,))

            # --- kinematics ---
            vx, vy, ax, ay = self._compute_kinematics(xs, ys)

            feats = np.stack([xs, ys, vx, vy, ax, ay], axis=1)  # (T, 6)

            # --- sliding windows ---
            for i in range(self.half, len(feats) - self.half):
                window = feats[i - self.half:i + self.half + 1].reshape(-1)
                X.append(window)
                y.append(LABEL_MAP[labels[i]])

            return np.array(X), np.array(y)




        

    
