import matplotlib.pyplot as plt
import numpy as np

# Data
iterations = ["Iter1", "Iter2", "Iter3", "Iter4", "Iter5", "Iter6"]
coverage = np.array([220, 224, 167, 151, 272, 147])
total = 1183
accuracy_g = np.array([94.758, 85.714, 80.838, 86.755, 67.279, 74.150])

# Compute proportional accuracies
prop_accuracies = (coverage / total) * (accuracy_g / 100)

# Colors for each iteration
colors = ["#E24A33", "#348ABD", "#988ED5", "#777777", "#FBC15E", "#8EBA42"]

# Total proportional accuracy (target height)
total_acc = 0.88

# Create figure
fig, ax = plt.subplots(figsize=(8, 5))
bottom = np.zeros(len(iterations))

# Plot cumulative stacks
for i in range(len(iterations)):
    ax.bar(iterations[i:], prop_accuracies[i], bottom=bottom[i:], 
           color=colors[i], edgecolor="white", label=f"Iter {i+1}")
    bottom[i:] += prop_accuracies[i]

# --- Add black residual segment ---
residual = total_acc - bottom  # space left to fill up to dashed line
ax.bar(iterations, residual, bottom=bottom, color="black", edgecolor="white", label="Residual")

# --- Add dashed purple line at total accuracy ---
ax.axhline(y=total_acc, color="purple", linestyle="--", linewidth=1.5)
ax.text(len(iterations) - 0.8, total_acc + 0.005, f"Total = {total_acc:.2f}", color="purple")

# --- Formatting ---
ax.set_ylim(0, total_acc + 0.05)
ax.set_ylabel("Proportional Accuracy")
ax.set_xlabel("Iterations")
ax.set_title("Cumulative Proportional Accuracy Across Iterations (CUB-200 VIT)")
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Iteration")

plt.tight_layout()
plt.savefig("cumulative_barchart_with_residual.png", dpi=150, bbox_inches="tight")
plt.show(block=True)
