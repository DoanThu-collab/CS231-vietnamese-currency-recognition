import os

def count_per_class(folder_path):
    class_counts = {}

    for class_name in sorted(os.listdir(folder_path)):
        class_path = os.path.join(folder_path, class_name)

        if not os.path.isdir(class_path):
            continue

        images = [
            f for f in os.listdir(class_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ]

        class_counts[class_name] = len(images)

    return class_counts


def print_counts(split_name, class_counts):
    print(f"\n===== {split_name.upper()} =====")
    total = 0

    for cls, count in class_counts.items():
        print(f"{cls:10s}: {count}")
        total += count

    print(f"Total {split_name}: {total}")
    return total


# ================= MAIN =================
dataset_path = "data"

train_counts = count_per_class(os.path.join(dataset_path, "train"))
val_counts   = count_per_class(os.path.join(dataset_path, "val"))
test_counts  = count_per_class(os.path.join(dataset_path, "test"))

train_total = print_counts("train", train_counts)
val_total   = print_counts("val", val_counts)
test_total  = print_counts("test", test_counts)

print("\n===== OVERALL =====")
print(f"Train: {train_total}")
print(f"Validation: {val_total}")
print(f"Test: {test_total}")
print(f"Total: {train_total + val_total + test_total}")