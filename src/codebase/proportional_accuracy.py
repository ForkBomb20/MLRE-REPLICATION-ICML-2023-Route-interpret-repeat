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

# --- NEW: correct residual proportional accuracy ---
residual_coverage = np.array([963, 739, 572, 421, 149, 2])
accuracy_r = np.array([85.670, 84.574, 84.615, 81.948, 83.221, 100.0])
residual = (residual_coverage / total) * (accuracy_r / 100)

# Create figure
fig, ax = plt.subplots(figsize=(8, 5))
bottom = np.zeros(len(iterations))

# Plot cumulative stacks
for i in range(len(iterations)):
    ax.bar(iterations[i:], prop_accuracies[i], bottom=bottom[i:], 
           color=colors[i], edgecolor="white", label=f"Iter {i+1}")
    bottom[i:] += prop_accuracies[i]

# --- Add TRUE residual segment ---
ax.bar(iterations, residual, bottom=bottom, color="black", edgecolor="white", label="Residual")

# --- Keep the dashed purple line ---
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
