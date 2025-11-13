import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Extract unique class-explanation combinations from CSV file')
parser.add_argument('--explanations_file', type=str, help='Path to the CSV file containing predictions and explanations')
parser.add_argument('--classes_file', type=str, help='Path to the file containing int to class name mappings')
parser.add_argument('--expert', type=int, help='Expert number (e.g., 1, 2, 3)')
args = parser.parse_args()

df = pd.read_csv(args.explanations_file)

classes_dict = {}
with open(args.classes_file, 'r') as f:
    for line in f:
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            class_num = int(parts[0])
            class_name = parts[1]
            classes_dict[class_num] = class_name

def parse_conjunction_to_set(expr_str):
    """
    Convert a conjunction string like
    'A & B & ~C' → {'A', 'B', '~C'}
    """
    expr_str = expr_str.strip().replace("(", "").replace(")", "")
    if not expr_str:
        return set()
    parts = [p.strip() for p in expr_str.split("&") if p.strip()]
    return set(parts)

def structural_or_simplify(expressions):
    """
    Combine multiple AND-only expressions (as strings) with OR,
    and factor out common literals.

    Example:
      ["A & B & C", "A & B & ~C"] → "(A & B) & (C | ~C)"
    """
    if len(expressions) == 1:
        return expressions[0]

    sets = [parse_conjunction_to_set(e) for e in expressions]
    # Find intersection (shared literals across all explanations)
    common = set.intersection(*sets) if sets else set()

    # Build remainder for each unique combination
    remainder = []
    for s in sets:
        diff = s - common
        if diff:
            remainder.append(" & ".join(sorted(diff)))
        else:
            remainder.append("TRUE")  # covers the 'no difference' case

    # Reconstruct factored expression
    if common and remainder:
        common_str = " & ".join(sorted(common))
        remainder_str = " | ".join(f"({r})" for r in remainder)
        return f"({common_str}) & ({remainder_str})"
    elif common:
        return " & ".join(sorted(common))
    else:
        return " | ".join(f"({r})" for r in remainder)

unique_combinations = (
    df[['g_pred', 'actual_explanations']]
    .drop_duplicates()
    .sort_values(by=['g_pred', 'actual_explanations'])
)

# Group explanations by class
class_explanations = {}
for _, row in unique_combinations.iterrows():
    g_pred = row['g_pred']
    explanation = row['actual_explanations']
    class_explanations.setdefault(g_pred, []).append(explanation)

# Apply structural simplification
simplified_by_class = {}
for g_pred, explanations in class_explanations.items():
    try:
        simplified_by_class[g_pred] = structural_or_simplify(explanations)
    except Exception as e:
        simplified_by_class[g_pred] = " | ".join(f"({exp})" for exp in explanations)
        print(f"Warning: simplification failed for class {g_pred}: {e}")

output_file = f"unique_class_explanations_expert_{args.expert}.txt"
with open(output_file, 'w') as f:
    f.write(f"Unique (Class Prediction, Explanation) Combinations - Expert {args.expert}\n")
    f.write("=" * 80 + "\n\n")

    f.write("SIMPLIFIED COMBINED EXPLANATIONS (OR logic):\n")
    f.write("=" * 80 + "\n\n")

    for g_pred in sorted(simplified_by_class.keys()):
        class_desc = classes_dict.get(g_pred, "Unknown class")
        simplified_explanation = simplified_by_class[g_pred]

        f.write(f"Class: {g_pred} - {class_desc}\n")
        f.write(f"Combined Explanation: {simplified_explanation}\n")
        f.write("-" * 80 + "\n\n")

print(f"Unique combinations written to {output_file}")
print(f"Total classes with explanations: {len(simplified_by_class)}")
print(f"\nUnique explanations per class:")
print(df.groupby('g_pred')['actual_explanations'].nunique())
