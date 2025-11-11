import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.functional import one_hot
import sklearn.metrics as metrics
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("./codebase"))
from Explainer.loss_F import loss_fn_kd, entropy_loss, Selective_Distillation_Loss
from Explainer.models.Gated_Logic_Net import Gated_Logic_Net
from Explainer.models.explainer import Explainer
from Explainer.models.pi import Pi
from dataset.dataset_cubs import Dataset_cub_for_explainer
from dataset.utils_dataset import get_dataset_with_image_and_attributes
from Explainer.models.concepts import Conceptizator

root = "/scratch/eecs498f25s007_class_root/eecs498f25s007_class/shared_data/group12/out/cub"
experiment = (
    "explainer/ResNet101/"
    "lr_0.01_epochs_120_temperature-lens_0.7_use-concepts-as-pi-input_True_"
    "input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_"
    "lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_"
    "layer_layer4_explainer_init_none"
)

iterations = [f"iter{i}" for i in range(1, 7)]
expert_type = "explainer"
output = "g_outputs"
total_samples = 1183

test_tensor_preds = []
test_tensor_y = []
test_tensor_preds_bb = []

for it in iterations:
    print(f"Loading {it}...")
    test_tensor_preds.append(
        torch.load(
            os.path.join(
                root, experiment, "cov_0.2_lr_0.01", it, expert_type, output, "test_tensor_preds.pt"
            )
        )
    )
    test_tensor_y.append(
        torch.load(
            os.path.join(
                root, experiment, "cov_0.2_lr_0.01", it, expert_type, output, "test_tensor_y.pt"
            )
        )
    )
    test_tensor_preds_bb.append(
        torch.load(
            os.path.join(
                root, experiment, "cov_0.2_lr_0.01", it, expert_type, output, "test_tensor_preds_bb.pt"
            )
        )
    )

# -----------------------
# Compute coverage ratios
# -----------------------
sizes = [t.size(0) for t in test_tensor_preds]
cumulative = np.cumsum(sizes)
ratios_conditional = [
    sizes[0] / total_samples,
    sizes[1] / (total_samples - sizes[0]),
    sizes[2] / (total_samples - sizes[0] - sizes[1]),
    sizes[3] / (total_samples - sizes[0] - sizes[1] - sizes[2]),
    sizes[4] / (total_samples - sizes[0] - sizes[1] - sizes[2] - sizes[3]),
    sizes[5] / (total_samples - sizes[0] - sizes[1] - sizes[2] - sizes[3] - sizes[4]),
]

# Actual overall coverage (relative to total)
ratios_total = [s / total_samples for s in sizes]
residual = 1 - sum(ratios_total)
ratios_total.append(max(residual, 0.0))  # add residual

print("Expert sample sizes:", sizes)
print("Coverage ratios (over total):", ratios_total)

# -----------------------
# Plot coverage
# -----------------------
category_names = [
    "Expert1", "Expert2", "Expert3", "Expert4", "Expert5", "Expert6", "Residual"
]
results = {"ResNet101": ratios_total}

def survey(results, category_names):
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = [
        "darkorange", "tab:blue", "tab:green", "tab:cyan", "tab:purple", "tab:red", "black"
    ]

    fig, ax = plt.subplots(figsize=(9, 2))
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        ax.barh(labels, widths, left=starts, height=0.5, label=colname, color=color)

    ax.legend(
        ncol=len(category_names),
        bbox_to_anchor=(0, 1),
        loc="lower left",
        fontsize="small",
    )
    ax.set_xlabel("Coverage ratio over total test samples")
    plt.tight_layout()
    return fig, ax

survey(results, category_names)
plt.show()