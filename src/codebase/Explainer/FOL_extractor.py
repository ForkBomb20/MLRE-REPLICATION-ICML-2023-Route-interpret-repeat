import pandas as pd
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Extract unique class-explanation combinations from CSV file')
parser.add_argument('--explanations_file', type=str, help='Path to the CSV file containing predictions and explanations')
parser.add_argument('--classes_file', type=str, help='Path to the file containing int to class name mappings')
parser.add_argument('--expert', type=int, help='Expert number (e.g., 1, 2, 3)')
args = parser.parse_args()

# Read the CSV file with predictions and explanations
df = pd.read_csv(args.explanations_file)

# Read the classes file
classes_dict = {}
with open(args.classes_file, 'r') as f:
    for line in f:
        parts = line.strip().split(maxsplit=1)
        if len(parts) == 2:
            class_num = int(parts[0])
            class_name = parts[1]
            classes_dict[class_num] = class_name

# Get unique combinations of g_pred and actual_explanations
unique_combinations = df[['g_pred', 'actual_explanations']].drop_duplicates().sort_values(by=['g_pred', 'actual_explanations'])

# Write to output file
output_file = f"unique_class_explanations_expert_{args.expert}.txt"
with open(output_file, 'w') as f:
    f.write(f"Unique (Class Prediction, Explanation) Combinations - Expert {args.expert}\n")
    f.write("=" * 80 + "\n\n")
    
    for _, row in unique_combinations.iterrows():
        g_pred = row['g_pred']
        explanation = row['actual_explanations']
        
        # Get class description
        class_desc = classes_dict.get(g_pred, "Unknown class")
        
        f.write(f"Class: {g_pred} - {class_desc}\n")
        f.write(f"Explanation: {explanation}\n")
        f.write("-" * 80 + "\n\n")

print(f"Unique combinations written to {output_file}")
print(f"Total unique combinations: {len(unique_combinations)}")
print(f"\nUnique explanations per class:")
print(df.groupby("g_pred")["actual_explanations"].nunique())