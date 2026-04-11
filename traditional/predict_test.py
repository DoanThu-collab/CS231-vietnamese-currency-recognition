import cv2
import numpy as np
import os, pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

ORB             = cv2.ORB_create(nfeatures=500)
BF              = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
DISTANCE_THRESH = 50


# ─────────────────────────────────────────────
# STEP 1: Predict 1 ảnh
# ─────────────────────────────────────────────
def predict_one(image_path, templates):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "unknown"
    img = cv2.resize(img, (224, 224))

    _, des = ORB.detectAndCompute(img, None)

    # ✅ Fix: blur nhẹ rồi thử lại nếu không detect được
    if des is None:
        img = cv2.GaussianBlur(img, (3, 3), 0)
        _, des = ORB.detectAndCompute(img, None)

    if des is None:
        return "unknown"

    scores = {}
    for class_name, tmpl_matrix in templates.items():
        try:
            matches = BF.match(des, tmpl_matrix)
            good    = [m for m in matches if m.distance < DISTANCE_THRESH]
            scores[class_name] = len(good)
        except Exception:
            scores[class_name] = 0

    return max(scores, key=scores.get)


# ─────────────────────────────────────────────
# STEP 2: Evaluate toàn bộ test set
# ─────────────────────────────────────────────
def evaluate_test(test_dir, templates):
    print("Evaluating on test set...")

    y_true, y_pred = [], []

    for class_name in sorted(os.listdir(test_dir)):
        class_path = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        class_correct = 0
        class_total   = 0

        for img_file in sorted(os.listdir(class_path)):
            path = os.path.join(class_path, img_file)
            pred = predict_one(path, templates)

            is_correct     = (pred == class_name)
            class_correct += int(is_correct)
            class_total   += 1

            y_true.append(class_name)
            y_pred.append(pred)

        class_acc = class_correct / class_total if class_total > 0 else 0
        print(f"  [{class_name}] {class_correct:3d}/{class_total:3d} "
              f"→ {class_acc:.2%} {'✅' if class_acc >= 0.9 else '⚠️'}")

    return y_true, y_pred


# ─────────────────────────────────────────────
# STEP 3: Lưu report dạng ảnh PNG
# ─────────────────────────────────────────────
def save_report(y_true, y_pred, val_acc=0.9293,
                save_path="report/assets/orb_results.png"):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # ✅ Fix: lấy classes từ cả y_true lẫn y_pred
    classes = sorted(list(set(y_true) | set(y_pred)))

    # ✅ Fix: truyền labels + zero_division để tránh lỗi
    report   = classification_report(
        y_true, y_pred,
        labels=classes,
        target_names=classes,
        output_dict=True,
        zero_division=0
    )
    test_acc = accuracy_score(y_true, y_pred)
    cm       = confusion_matrix(y_true, y_pred, labels=classes)

    # ── Layout 2×2 ───────────────────────────────
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor("#0d1117")
    fig.suptitle("ORB Feature Matching — Test Set Results",
                 fontsize=18, fontweight="bold", color="white", y=0.98)

    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])  # Overall metrics
    ax2 = fig.add_subplot(gs[0, 1])  # Val vs Test
    ax3 = fig.add_subplot(gs[1, 0])  # Per-class F1
    ax4 = fig.add_subplot(gs[1, 1])  # Confusion matrix

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_facecolor("#161b22")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")
        ax.tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    # ── Chart 1: Overall metrics ─────────────────
    metrics = ["Precision\n(macro)", "Recall\n(macro)", "F1\n(macro)", "Accuracy"]
    values  = [
        report["macro avg"]["precision"],
        report["macro avg"]["recall"],
        report["macro avg"]["f1-score"],
        test_acc
    ]
    colors = ["#58a6ff", "#3fb950", "#f78166", "#e3b341"]
    bars   = ax1.bar(metrics, values, color=colors, width=0.5, zorder=3)
    ax1.set_ylim(0, 1.15)
    ax1.set_title("Overall Metrics", fontweight="bold")
    ax1.yaxis.grid(True, color="#30363d", zorder=0)
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.02,
                 f"{val:.2%}", ha="center", va="bottom",
                 color="white", fontsize=11, fontweight="bold")

    # ── Chart 2: Val vs Test ─────────────────────
    labels = ["Val Accuracy", "Test Accuracy"]
    accs   = [val_acc, test_acc]
    cols   = ["#58a6ff", "#3fb950"]
    bars2  = ax2.bar(labels, accs, color=cols, width=0.4, zorder=3)
    ax2.set_ylim(0, 1.15)
    ax2.set_title("Val vs Test Accuracy", fontweight="bold")
    ax2.yaxis.grid(True, color="#30363d", zorder=0)
    for bar, val in zip(bars2, accs):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.02,
                 f"{val:.2%}", ha="center", va="bottom",
                 color="white", fontsize=13, fontweight="bold")
    diff   = abs(test_acc - val_acc)
    status = "✅ Stable" if diff <= 0.05 else "⚠️ Check"
    ax2.text(0.5, 0.12, f"Diff: {diff:.2%}  {status}",
             ha="center", transform=ax2.transAxes,
             color="#e3b341", fontsize=11)

    # ── Chart 3: Per-class F1 ────────────────────
    # ✅ Fix: bỏ "unknown" ra khỏi chart nếu có
    chart_classes = [c for c in classes if c != "unknown"]
    f1_scores     = [report[c]["f1-score"] for c in chart_classes]
    bar_cols      = ["#3fb950" if f >= 0.9 else
                     "#e3b341" if f >= 0.75 else
                     "#f78166" for f in f1_scores]
    bars3 = ax3.barh(chart_classes, f1_scores, color=bar_cols, zorder=3)
    ax3.set_xlim(0, 1.15)
    ax3.set_title("F1-Score per Class", fontweight="bold")
    ax3.xaxis.grid(True, color="#30363d", zorder=0)
    ax3.axvline(x=0.9, color="#58a6ff", linestyle="--",
                linewidth=1, alpha=0.7)
    for bar, val in zip(bars3, f1_scores):
        ax3.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{val:.2f}", va="center", color="white", fontsize=9)
    legend_patches = [
        mpatches.Patch(color="#3fb950", label="F1 ≥ 0.90"),
        mpatches.Patch(color="#e3b341", label="F1 ≥ 0.75"),
        mpatches.Patch(color="#f78166", label="F1 < 0.75"),
    ]
    ax3.legend(handles=legend_patches, loc="lower right",
               facecolor="#161b22", edgecolor="#30363d",
               labelcolor="white", fontsize=8)

    # ── Chart 4: Confusion matrix ─────────────────
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes,
                ax=ax4, cbar=True,
                linewidths=0.5, linecolor="#30363d")
    ax4.set_title("Confusion Matrix", fontweight="bold")
    ax4.set_xlabel("Predicted", color="white")
    ax4.set_ylabel("True",      color="white")
    ax4.tick_params(axis="x", rotation=45, colors="white")
    ax4.tick_params(axis="y", rotation=0,  colors="white")

    plt.savefig(save_path, dpi=150,
                bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"Saved → {save_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    TEST_DIR = "data/test"
    VAL_ACC  = 0.9293   # ← thay bằng val accuracy thực tế của bạn

    # Load model
    with open("traditional/orb_model.pkl", "rb") as f:
        templates = pickle.load(f)
    print("Loaded → traditional/orb_model.pkl\n")

    # Evaluate
    y_true, y_pred = evaluate_test(TEST_DIR, templates)

    # In terminal
    test_acc = accuracy_score(y_true, y_pred)

    # ✅ Fix: dùng labels + zero_division khi in terminal
    classes = sorted(list(set(y_true) | set(y_pred)))
    print(f"\nTest Accuracy: {test_acc:.2%}")
    print(classification_report(
        y_true, y_pred,
        labels=classes,
        target_names=classes,
        zero_division=0
    ))

    # Lưu ảnh report
    save_report(y_true, y_pred,
                val_acc=VAL_ACC,
                save_path="report/assets/orb_results.png")

    # So sánh val vs test
    diff = abs(test_acc - VAL_ACC)
    print("\n" + "=" * 40)
    print(f"Val  Accuracy : {VAL_ACC:.2%}")
    print(f"Test Accuracy : {test_acc:.2%}")
    print(f"Chênh lệch    : {diff:.2%} "
          f"{'→ Stable ✅' if diff <= 0.05 else '→ Check ⚠️'}")
    print("=" * 40)
    
    
    
    
    
    