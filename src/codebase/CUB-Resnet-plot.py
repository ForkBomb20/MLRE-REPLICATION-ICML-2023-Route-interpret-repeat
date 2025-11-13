import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath("./codebase"))


# root = "/scratch/eecs498f25s007_class_root/eecs498f25s007_class/shared_data/group12/out/cub"
# experiment = (
#     "explainer/ResNet101/"
#     "lr_0.01_epochs_120_temperature-lens_0.7_use-concepts-as-pi-input_True_"
#     "input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_"
#     "lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_"
#     "layer_layer4_explainer_init_none"
# )

# iterations = [f"iter{i}" for i in range(1, 7)]
# expert_type = "explainer"
# output = "g_outputs"
# total_samples = 1183

# test_tensor_preds = []
# test_tensor_y = []
# test_tensor_preds_bb = []

# for it in iterations:
#     print(f"Loading {it}...")
#     test_tensor_preds.append(
#         torch.load(
#             os.path.join(
#                 root, experiment, "cov_0.2_lr_0.01", it, expert_type, output, "test_tensor_preds.pt"
#             )
#         )
#     )
#     test_tensor_y.append(
#         torch.load(
#             os.path.join(
#                 root, experiment, "cov_0.2_lr_0.01", it, expert_type, output, "test_tensor_y.pt"
#             )
#         )
#     )
#     test_tensor_preds_bb.append(
#         torch.load(
#             os.path.join(
#                 root, experiment, "cov_0.2_lr_0.01", it, expert_type, output, "test_tensor_preds_bb.pt"
#             )
#         )
#     )

# # -----------------------
# # Compute coverage ratios
# # -----------------------
# sizes = [t.size(0) for t in test_tensor_preds]
# cumulative = np.cumsum(sizes)
# ratios_conditional = [
#     sizes[0] / total_samples,
#     sizes[1] / (total_samples - sizes[0]),
#     sizes[2] / (total_samples - sizes[0] - sizes[1]),
#     sizes[3] / (total_samples - sizes[0] - sizes[1] - sizes[2]),
#     sizes[4] / (total_samples - sizes[0] - sizes[1] - sizes[2] - sizes[3]),
#     sizes[5] / (total_samples - sizes[0] - sizes[1] - sizes[2] - sizes[3] - sizes[4]),
# ]

# # Actual overall coverage (relative to total)
# ratios_total = [s / total_samples for s in sizes]
# residual = 1 - sum(ratios_total)
# ratios_total.append(max(residual, 0.0))  # add residual

# print("Expert sample sizes:", sizes)
# print("Coverage ratios (over total):", ratios_total)

# # -----------------------
# # Plot coverage
# # -----------------------
# category_names = [
#     "Expert1", "Expert2", "Expert3", "Expert4", "Expert5", "Expert6", "Residual"
# ]
# results = {"ResNet101": ratios_total}

# def survey(results, category_names):
#     labels = list(results.keys())
#     data = np.array(list(results.values()))
#     data_cum = data.cumsum(axis=1)
#     category_colors = [
#         "darkorange", "tab:blue", "tab:green", "tab:cyan", "tab:purple", "tab:red", "black"
#     ]

#     fig, ax = plt.subplots(figsize=(9, 2))
#     ax.set_xlim(0, np.sum(data, axis=1).max())

#     for i, (colname, color) in enumerate(zip(category_names, category_colors)):
#         widths = data[:, i]
#         starts = data_cum[:, i] - widths
#         ax.barh(labels, widths, left=starts, height=0.5, label=colname, color=color)

#     ax.legend(
#         ncol=len(category_names),
#         bbox_to_anchor=(0, 1),
#         loc="lower left",
#         fontsize="small",
#     )
#     ax.set_xlabel("Coverage ratio over total test samples")
#     plt.tight_layout()
#     return fig, ax

# survey(results, category_names)
# plt.show()


def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def counterfactual_concept_counts(n_experts, path):
    all_counts = {}
    
    for i in range(1, n_experts + 1):
        iteration_path = os.path.join(path, f"iter{i}", "explainer", "g_outputs")
        concept_file = os.path.join(iteration_path, "test_tensor_concepts.pt")
        tensor_alpha_norm_file = os.path.join(iteration_path, "test_tensor_alpha_norm.pt")
     
        if not os.path.exists(concept_file) or not os.path.exists(tensor_alpha_norm_file):
            print(f"Skipping Expert {i}, missing files in {iteration_path}")
            continue

        # Load tensors
        test_tensor_concepts = torch.load(concept_file)
        tensor_alpha_norm = torch.load(tensor_alpha_norm_file)
        x_to_bool = 0.5
        test_tensor_concepts_bool = (test_tensor_concepts.cpu() > x_to_bool).to(torch.float)

        # Compute number of concepts used per sample
        concepts_per_sample = test_tensor_concepts_bool.sum(dim=1).tolist()
        avg_concepts = sum(concepts_per_sample) / len(concepts_per_sample)

        print(f"<<<<<<< Expert : {i} >>>>>>>>")
        print(f"Expert {i} average concepts used: {avg_concepts}")
        print(f"Expert {i} concept counts per sample: {concepts_per_sample[:10]} ...")  # show first 10

        all_counts[i] = {
            "avg_concepts": avg_concepts,
            "counts_per_sample": concepts_per_sample
        }

    return all_counts

def plot_concept_counts_boxplot(concept_counts_all):
    """
    concept_counts_all: dict where keys = 1..6 (experts), values = dict with
                        "counts_per_sample" as list of concept counts per sample
    """
    # Prepare data
    data_for_boxplot = [concept_counts_all[i]["counts_per_sample"] for i in range(1, 7)]

    plt.figure(figsize=(10, 6))
    plt.boxplot(data_for_boxplot, labels=[f"Expert {i}" for i in range(1, 7)], patch_artist=True)
    plt.xlabel("Expert")
    plt.ylabel("Number of Concepts Used")
    plt.title("Distribution of Concepts Used per Sample Across Experts")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("concept_counts_boxplot.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    lr = 0.01
    cov = 0.2
    path = "/scratch/eecs498f25s007_class_root/eecs498f25s007_class/shared_data/group12/out/cub/explainer/ResNet101/lr_0.01_epochs_120_temperature-lens_0.7_use-concepts-as-pi-input_True_input-size-pi_2048_cov_0.2_alpha_0.5_selection-threshold_0.5_lambda-lens_0.0001_alpha-KD_0.9_temperature-KD_10.0_hidden-layers_1_layer_layer4_explainer_init_none/cov_0.2_lr_0.01"
    n_experts = 6
    concept_counts_all = counterfactual_concept_counts(n_experts, path)
    for i in range(1, n_experts + 1):
        counts = concept_counts_all[i]["counts_per_sample"]
        avg = np.mean(counts)
        print(f"<<<<<<< Expert : {i} >>>>>>>>")
        print(f"Expert {i} average concepts used: {avg}")
        print(f"Expert {i} concept counts per sample: {counts[:10]} ...") 

    plot_concept_counts_boxplot(concept_counts_all)