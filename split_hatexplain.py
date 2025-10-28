import json
import os
from collections import Counter, defaultdict

# === CONFIG ===
uniqname = os.getenv("USER")  # Adjust as needed for your environment
INPUT_FILE = "/scratch/eecs498f25s007_class_root/eecs498f25s007_class/shared_data/group12/data/hatexplain/dataset.json"

OUTPUT_BASE_DIR = f"/scratch/eecs498f25s007_class_root/eecs498f25s007_class/{uniqname}/MLRE-REPLICATION-ICML-2023-Route-interpret-repeat/data/hatexplain"

LABEL_FOLDERS = ["offensive", "hatespeech", "normal"]
for folder in LABEL_FOLDERS:
    os.makedirs(os.path.join(OUTPUT_BASE_DIR, folder), exist_ok=True)

# === LOAD DATA ===
with open(INPUT_FILE, "r") as f:
    data = json.load(f)

print(f"Loaded {len(data)} posts.")

# === TRACKING COUNTS ===
saved_counts = defaultdict(int)
skipped = 0

# === PROCESS EACH POST ===
for post_id, post_data in data.items():
    labels = [ann["label"] for ann in post_data.get("annotators", [])]
    label_counts = Counter(labels)

    # Skip posts where all annotators disagree (no majority)
    if len(label_counts) == len(labels) and len(labels) > 1:
        skipped += 1
        continue

    if label_counts:
        # Determine majority label
        majority_label = label_counts.most_common(1)[0][0]
    else:
        majority_label = "normal"

    if majority_label not in LABEL_FOLDERS:
        majority_label = "normal"

    # Save post
    output_path = os.path.join(OUTPUT_BASE_DIR, majority_label, f"{post_id}.json")
    with open(output_path, "w") as out_f:
        json.dump({post_id: post_data}, out_f, indent=2)

    saved_counts[majority_label] += 1

# === SUMMARY OUTPUT ===
print("\n✅ Finished splitting dataset.")
total_saved = sum(saved_counts.values())
print(f"  Total saved: {total_saved}")
for label in LABEL_FOLDERS:
    print(f"    {label}: {saved_counts[label]} posts")
print(f"  Skipped (all annotators disagree): {skipped} posts")
