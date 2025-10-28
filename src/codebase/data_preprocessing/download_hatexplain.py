import os
import json
import numpy as np
from collections import defaultdict, Counter

def aggregate_rationales(rationales):
    """
    Safely aggregate rationales from annotators.
    rationales: list of lists, each list is one annotator's rationale (bool or 0/1)
    Returns 1D numpy array of aggregated rationale
    """
    if not rationales:
        return None

    if len(rationales) == 1:
        return np.array(rationales[0], dtype=bool)


    max_len = max(len(r) for r in rationales)
    padded = [
        r + [0]*(max_len - len(r)) if len(r) < max_len else r
        for r in rationales
    ]
    rationales_array = np.array(padded, dtype=int)


    aggregated = np.sum(rationales_array, axis=0) >= 2
    return aggregated

def format_hatexplain_for_moie(
    hatexplain_json_path,
    post_id_divisions_path,
    output_dir
):
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 70)
    print("Loading HateXplain data")
    print("=" * 70)
    
    with open(hatexplain_json_path, 'r') as f:
        hatexplain_data = json.load(f)
    
    with open(post_id_divisions_path, 'r') as f:
        post_divisions = json.load(f)
    
    all_post_ids = post_divisions['train'] + post_divisions['val'] + post_divisions['test']
    print(f"Total posts: {len(all_post_ids)}")
    
    # STEP 1: Creating label array
    label_map = {'hatespeech': 0, 'offensive': 1, 'normal': 2}
    label_names = ['hatespeech', 'offensive', 'normal']
    labels = []
    post_id_to_idx = {}
    
    for idx, post_id in enumerate(all_post_ids):
        post = hatexplain_data[post_id]
        annotator_labels = [ann['label'] for ann in post['annotators']]
        majority_label = Counter(annotator_labels).most_common(1)[0][0]
        labels.append(label_map[majority_label])
        post_id_to_idx[post_id] = idx
    labels = np.array(labels, dtype=np.int64)
    
    print("Label distribution:")
    for i, name in enumerate(label_names):
        count = np.sum(labels == i)
        print(f"  {i}.{name}: {count} ({count/len(labels)*100:.1f}%)")
    
    # STEP 2: Extracting candidate concepts
    all_targets = set()
    for post_id in all_post_ids:
        post = hatexplain_data[post_id]
        for ann in post['annotators']:
            all_targets.update(ann['target'])
    all_targets = sorted(list(all_targets))
    print(f"Target concepts: {len(all_targets)}")
    print(f"Examples: {all_targets[:5]}")
    
    token_rationale_counts = defaultdict(int)
    token_total_counts = defaultdict(int)
    
    for post_id in all_post_ids:
        post = hatexplain_data[post_id]
        tokens = post['post_tokens']
        rationales = post['rationales']
        
        aggregated_rationale = aggregate_rationales(rationales)
        if aggregated_rationale is None:
            continue
        
        for token_idx, token in enumerate(tokens):
            token_lower = token.lower()
            token_total_counts[token_lower] += 1
            if token_idx < len(aggregated_rationale) and aggregated_rationale[token_idx]:
                token_rationale_counts[token_lower] += 1
    
    MIN_RATIONALE_FREQ = 20
    MIN_SELECTIVITY = 0.4
    candidate_tokens = [
        token for token, rat_count in token_rationale_counts.items()
        if rat_count >= MIN_RATIONALE_FREQ and rat_count / token_total_counts[token] >= MIN_SELECTIVITY
    ]
    candidate_tokens = sorted(candidate_tokens)
    print(f"Token concepts: {len(candidate_tokens)}")
    print(f"Examples: {candidate_tokens[:5]}")
    
    # STEP 3: Creating concept matrix
    N = len(all_post_ids)
    n_targets = len(all_targets)
    n_tokens = len(candidate_tokens)
    temp_original = np.zeros((N, n_targets + n_tokens), dtype=np.float32)
    
    for i, post_id in enumerate(all_post_ids):
        post = hatexplain_data[post_id]
        
        # Target concepts
        target_counts = Counter()
        for ann in post['annotators']:
            target_counts.update(ann['target'])
        for target, count in target_counts.items():
            if count >= 2 or len(post['annotators']) == 1:  # single annotator also counts
                target_idx = all_targets.index(target)
                temp_original[i, target_idx] = 1
        
        # Token concepts
        tokens = post['post_tokens']
        rationales = post['rationales']
        aggregated_rationale = aggregate_rationales(rationales)
        if aggregated_rationale is None:
            continue
        
        for token_idx, token in enumerate(tokens):
            token_lower = token.lower()
            if token_lower in candidate_tokens:
                token_concept_idx = candidate_tokens.index(token_lower) + n_targets
                if token_idx < len(aggregated_rationale) and aggregated_rationale[token_idx]:
                    temp_original[i, token_concept_idx] = 1
    
    # STEP 4: Filtering sparse concepts
    concept_occurrence = np.sum(temp_original, axis=0)
    MIN_OCCURRENCE = 50
    keep_concepts_mask = concept_occurrence >= MIN_OCCURRENCE
    all_candidate_names = [f"has_target::{t}" for t in all_targets] + [f"has_token::{t}" for t in candidate_tokens]
    final_concept_names = [all_candidate_names[i] for i in range(len(all_candidate_names)) if keep_concepts_mask[i]]
    
    original_attributes = temp_original[:, keep_concepts_mask].astype(np.float32)
    
    # STEP 6: Denoising
    denoised_attributes = np.zeros_like(original_attributes)
    for class_label in [0, 1, 2]:
        class_mask = (labels == class_label)
        class_posts = original_attributes[class_mask]
        if np.sum(class_mask) > 0:
            concept_freq = np.mean(class_posts, axis=0)
            denoised_attributes[class_mask, :] = concept_freq > 0.3
    
    # STEP 7: Save post files
    for i, class_name in enumerate(label_names):
        class_dir = os.path.join(output_dir, f"{i}.{class_name}")
        os.makedirs(class_dir, exist_ok=True)
    
    for idx, post_id in enumerate(all_post_ids):
        post = hatexplain_data[post_id]
        label = labels[idx]
        class_name = label_names[label]
        filename = f"{post_id}.txt"
        class_dir = os.path.join(output_dir, f"{label}.{class_name}")
        filepath = os.path.join(class_dir, filename)
        content = {
            'post_id': post_id,
            'index': idx,
            'label': int(label),
            'label_name': class_name,
            'text': ' '.join(post['post_tokens']),
            'tokens': post['post_tokens']
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(content, f, ensure_ascii=False, indent=2)
    
    # STEP 8: Save core files
    with open(os.path.join(output_dir, 'attributes_names.txt'), 'w') as f:
        json.dump(final_concept_names, f, indent=2)
    np.save(os.path.join(output_dir, 'original_attributes.npy'), original_attributes)
    np.save(os.path.join(output_dir, 'attributes.npy'), denoised_attributes)
    
    print("Formatting complete")
    return output_dir

# ============================================================
if __name__ == "__main__":
    uniqname = os.getenv("USER")
    
    INPUT_FILE = "/scratch/eecs498f25s007_class_root/eecs498f25s007_class/shared_data/group12/data/hatexplain/dataset.json"
    POST_ID_DIVISIONS = "/scratch/eecs498f25s007_class_root/eecs498f25s007_class/shared_data/group12/data/hatexplain/post_id_divisions.json"
    #OUTPUT_BASE_DIR = f"/scratch/eecs498f25s007_class_root/eecs498f25s007_class/{uniqname}/MLRE-REPLICATION-ICML-2023-Route-interpret-repeat/data/hatexplain"
    OUTPUT_BASE_DIR = f"/home/xinyiade/MLRE/data/hatexplain"
    
    
    output_dir = format_hatexplain_for_moie(
        hatexplain_json_path=INPUT_FILE,
        post_id_divisions_path=POST_ID_DIVISIONS,
        output_dir=OUTPUT_BASE_DIR
    )