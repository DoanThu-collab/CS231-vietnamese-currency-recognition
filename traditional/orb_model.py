import cv2
import numpy as np
import os
import pickle
import json
from sklearn.metrics import accuracy_score, classification_report

ORB = cv2.ORB_create(nfeatures=500)
BF  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

MAX_DESC_PER_CLASS = 50000  # giới hạn tối đa descriptors mỗi class


# ─────────────────────────────────────────────
# STEP 1: Build templates
# ─────────────────────────────────────────────
def build_templates(train_dir):
    print("Building templates...")
    templates = {}

    for class_name in sorted(os.listdir(train_dir)):
        class_path = os.path.join(train_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        all_des = []

        for img_file in os.listdir(class_path):
            path = os.path.join(class_path, img_file)
            try:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (224, 224))
                _, des = ORB.detectAndCompute(img, None)
                if des is not None:
                    all_des.append(des)
            except Exception as e:
                print(f"  Skip: {path} | {e}")

        if not all_des:
            continue

        # Gộp thành 1 matrix
        combined = np.vstack(all_des)

        # ✅ Giới hạn số descriptors để tránh lỗi BFMatcher
        if combined.shape[0] > MAX_DESC_PER_CLASS:
            idx = np.random.choice(combined.shape[0],
                                   MAX_DESC_PER_CLASS,
                                   replace=False)
            combined = combined[idx]

        templates[class_name] = combined
        print(f"  [{class_name}] → {len(os.listdir(class_path))} ảnh "
              f"| {combined.shape[0]} descriptors")

    return templates


# ─────────────────────────────────────────────
# STEP 2: Predict 1 ảnh
# ─────────────────────────────────────────────
def predict_one(image_path, templates):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "unknown"
    img = cv2.resize(img, (224, 224))

    _, des = ORB.detectAndCompute(img, None)
    if des is None:
        return "unknown"

    scores = {}
    for class_name, tmpl_matrix in templates.items():
        try:
            matches = BF.match(des, tmpl_matrix)
            good    = [m for m in matches if m.distance < 50]
            scores[class_name] = len(good)
        except Exception:
            scores[class_name] = 0

    return max(scores, key=scores.get)


# ─────────────────────────────────────────────
# STEP 3: Evaluate
# ─────────────────────────────────────────────
def evaluate(dataset_dir, templates, split_name="val"):
    print(f"\nEvaluating on {split_name} set...")

    y_true, y_pred, results = [], [], []

    for class_name in sorted(os.listdir(dataset_dir)):
        class_path = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        for img_file in sorted(os.listdir(class_path)):
            path = os.path.join(class_path, img_file)
            pred = predict_one(path, templates)

            y_true.append(class_name)
            y_pred.append(pred)
            results.append({"image": path, "true": class_name, "pred": pred})
            print(f"  {img_file:30s} | true: {class_name} | pred: {pred} "
                  f"| {'✓' if pred == class_name else '✗'}")

    acc = accuracy_score(y_true, y_pred)
    print(f"\n{split_name} Accuracy: {acc:.2%}")
    print(classification_report(y_true, y_pred))

    return results


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    TRAIN_DIR = "data/train"
    VAL_DIR   = "data/val"

    templates = build_templates(TRAIN_DIR)

    os.makedirs("traditional", exist_ok=True)
    with open("traditional/orb_model.pkl", "wb") as f:
        pickle.dump(templates, f)
    print("Saved → traditional/orb_model.pkl")

    results = evaluate(VAL_DIR, templates, split_name="val")

    with open("traditional/val_predictions.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print("Saved → traditional/val_predictions.json")