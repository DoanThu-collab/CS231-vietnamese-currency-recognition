import cv2
import numpy as np
import os, pickle, json
from sklearn.metrics import accuracy_score

ORB = cv2.ORB_create(nfeatures=500)
BF  = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
MAX_DESC = 50000


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

        combined = np.vstack(all_des)
        if combined.shape[0] > MAX_DESC:
            idx      = np.random.choice(combined.shape[0], MAX_DESC, replace=False)
            combined = combined[idx]

        templates[class_name] = combined
        print(f"  [{class_name}] → {combined.shape[0]} descriptors")

    return templates


def predict_one(image_path, templates, distance_thresh):
    """Predict với threshold truyền vào"""
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
            # Lọc good matches theo threshold truyền vào
            good = [m for m in matches if m.distance < distance_thresh]
            scores[class_name] = len(good)
        except Exception:
            scores[class_name] = 0

    return max(scores, key=scores.get)


def evaluate_with_thresh(dataset_dir, templates, distance_thresh):
    """Evaluate toàn bộ val set với 1 threshold cụ thể"""
    y_true, y_pred = [], []

    for class_name in sorted(os.listdir(dataset_dir)):
        class_path = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        for img_file in sorted(os.listdir(class_path)):
            path = os.path.join(class_path, img_file)
            pred = predict_one(path, templates, distance_thresh)
            y_true.append(class_name)
            y_pred.append(pred)

    return accuracy_score(y_true, y_pred)


def tune_threshold(dataset_dir, templates):
    """Thử nhiều threshold, in bảng so sánh, trả về threshold tốt nhất"""
    # Danh sách threshold cần thử
    thresholds = [30, 40, 50, 60, 70, 80]

    print("\n" + "=" * 40)
    print("TUNING DISTANCE_THRESH")
    print("=" * 40)
    print(f"{'Threshold':>12} | {'Accuracy':>10}")
    print("-" * 40)

    best_thresh = None
    best_acc    = -1
    results     = {}

    for thresh in thresholds:
        acc = evaluate_with_thresh(dataset_dir, templates, thresh)
        results[thresh] = acc
        marker = " ← best" if acc > best_acc else ""

        # Cập nhật best
        if acc > best_acc:
            best_acc    = acc
            best_thresh = thresh

        print(f"{thresh:>12} | {acc:>9.2%}{marker}")

    print("=" * 40)
    print(f"Best threshold : {best_thresh}")
    print(f"Best accuracy  : {best_acc:.2%}")

    return best_thresh, best_acc


def evaluate_final(dataset_dir, templates, best_thresh):
    """Evaluate lần cuối với threshold tốt nhất, lưu kết quả"""
    print(f"\nFinal evaluate với threshold = {best_thresh}...")
    y_true, y_pred, results = [], [], []

    for class_name in sorted(os.listdir(dataset_dir)):
        class_path = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        for img_file in sorted(os.listdir(class_path)):
            path = os.path.join(class_path, img_file)
            pred = predict_one(path, templates, best_thresh)

            y_true.append(class_name)
            y_pred.append(pred)
            results.append({
                "image": path,
                "true" : class_name,
                "pred" : pred
            })

    from sklearn.metrics import classification_report
    print(f"\nFinal Accuracy : {accuracy_score(y_true, y_pred):.2%}")
    print(classification_report(y_true, y_pred))

    # Lưu kết quả
    os.makedirs("traditional", exist_ok=True)
    with open("traditional/val_predictions.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print("Saved → traditional/val_predictions.json")


# ── MAIN ──────────────────────────────────────
if __name__ == "__main__":
    TRAIN_DIR = "data/train"
    VAL_DIR   = "data/val"

    # 1. Build templates
    templates = build_templates(TRAIN_DIR)

    os.makedirs("traditional", exist_ok=True)
    with open("traditional/orb_model.pkl", "wb") as f:
        pickle.dump(templates, f)
    print("Saved → traditional/orb_model.pkl")

    # 2. Tuning threshold trên val set
    best_thresh, best_acc = tune_threshold(VAL_DIR, templates)

    # 3. Evaluate lần cuối với threshold tốt nhất
    evaluate_final(VAL_DIR, templates, best_thresh)