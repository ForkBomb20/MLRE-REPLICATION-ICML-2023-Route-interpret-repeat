# HateXplain Dataset

## Original Location
https://github.com/hate-alert/HateXplain/blob/master/README.md

## Shared Location
```bash
/scratch/eecs498f25s007_class_root/eecs498f25s007_class/shared_data/group12/data/hatexplain
```
## Original Dataset Format
```bash
{
  "24198545_gab": {
    "post_id": "24198545_gab",
    "annotators": [
      {
        "label": "hatespeech",
        "annotator_id": 4,
        "target": ["African"]
      },
      {
        "label": "hatespeech",
        "annotator_id": 3,
        "target": ["African"]
      },
      {
        "label": "offensive",
        "annotator_id": 5,
        "target": ["African"]
      }
    ],
    "rationales":[
    [0,0,0,0,0,0,0,0,1,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    ],
    "post_tokens": ["and","this","is","why","i","end","up","with","nigger","trainee","doctors","who","can","not","speak","properly","lack","basic","knowledge","of","biology","it","truly","scary","if","the","public","only","knew"]
  }
}
```

## Copy and process dataset to local
```bash
python split_hatexplain.py #earlier version, only split the json into 3 subfolders

python download_hatexplain.py #current version, generating also txt/npy
```
## Folder Structure
hatexplain/
├── 0.hatespeech/
│   ├── 24198545.txt
│   ├── 24198546.txt
│   └── ...
├── 1.offensive/
│   ├── 24198601.txt
│   ├── 24198602.txt
│   └── ...
├── 2.normal/
│   ├── 24198701.txt
│   ├── 24198702.txt
│   └── ...
├── attributes_names.txt
├── original_attributes.npy
└── attributes.npy

## Features
- Aggregates rationales from multiple annotators
subject to change:
1.Token concepts are filtered by minimum frequency (MIN_RATIONALE_FREQ=20) and selectivity (MIN_SELECTIVITY=0.4)
2.Concept matrices are further filtered by occurrence (MIN_OCCURRENCE=50) and class-based denoising

- Extracts target concepts and token concepts based on rationales
- Generates two concept matrices:
  - `original_attributes.npy` — unprocessed concept matrix
  - `attributes.npy` — denoised concept matrix, filtered by class-specific frequency
- Saves the list of final concept names in `attributes_names.txt`

