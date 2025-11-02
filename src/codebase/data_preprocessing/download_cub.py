import os
import json
import numpy as np
import pandas as pd


def download_cub(root="CUB_200_2011", force=True):
    origin_dir = os.curdir
    os.chdir(root)

    os.system("mv ../attributes.txt .")
    os.rename("attributes.txt", "attributes_names.txt")
    
    # 312 attributes (rows), each row is attr_id - attr name
    with open("attributes_names.txt", "r") as f:
        attribute_names = []
        lines = f.readlines()
        for line in lines:
            attr_name = line.split(" ")[1]
            attribute_names.append(attr_name)

    # ============================================================================================
    ## Originial attributes ##
    # 11788 total images (rows), each row is image_id - class_id
    classes = pd.read_csv("image_class_labels.txt", sep=" ", header=None).to_numpy()[:, 1]
    print("Image_classes loaded")

    attributes = np.zeros(shape=(11788, 312))
    with open(os.path.join("attributes", "image_attribute_labels.txt"), "r") as f:
        lines = f.readlines()
        for line in lines:
            fields = line.split(" ")
            if fields[2] == "1":
                img_idx = int(fields[0]) - 1
                attr_idx = int(fields[1]) - 1
                attributes[img_idx][attr_idx] = 1

    # total images (11788) * total attributes (312), each cell is 1 if present, 0 if not based on human label
    np.save("original_attributes.npy", attributes)

    # ============================================================================================
    ## Denoised attributes ##
    # 200 lines (one class, same order as classes.txt), 312 space-separated cols (% of attribute in class, same order as attributes.txt)
    attribute_per_class = pd.read_csv(os.path.join("attributes", "class_attribute_labels_continuous.txt"), sep=" ",
                                      header=None).to_numpy()

    very_denoised_attributes = np.zeros(shape=(11788, 312))
    attribute_sparsity = np.zeros(attributes.shape[1])  # zeros of length 312
    for c in np.unique(classes):
        imgs = classes == c
        class_attributes = attribute_per_class[c - 1, :] > 50
        very_denoised_attributes[imgs, :] = class_attributes
        attribute_sparsity += class_attributes
    attributes_to_filter = attribute_sparsity < 10
    very_denoised_attributes = very_denoised_attributes[:, ~attributes_to_filter]

    # total images (11788) * total attributes (312), each cell is 1 if over 50% of that class has the attr, 0 if less; based on does the class have that attribute (class_attribute_labels)
    # Only the attributes that are present in >=10 classes are chosen
    # Same class has same list of attributes
    np.save("attributes.npy", very_denoised_attributes)

    # Filter attributes based on above criteria
    with open("attributes_names.txt", "w") as f:
        json.dump(np.asarray(attribute_names)[~attributes_to_filter].tolist(), f)

    # ============================================================================================
    # Clean up
    for item in os.listdir():
        if item not in ["images", "images.txt", "attributes_names.txt", "README",
                        "original_attributes.npy",  "attributes.npy"]:
            os.system(f"rm -r {item}")

    # Rename images
    os.system("mv images/* .")
    with open("images.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            fields = line.split(" ")
            img_idx = f"{int(fields[0]):05d}"
            img_name = fields[1][:-1]
            img_name = os.path.join(os.path.dirname(img_name), os.path.basename(img_name))
            new_name = os.path.join(os.path.dirname(img_name), img_idx) + os.path.splitext(img_name)[1]
            os.rename(img_name, new_name)
    os.system("rm -r images")
    os.remove("images.txt")
    os.system("mv ../CUB_200_2011 ../data")

    os.chdir(origin_dir)
    print("Dataset configured correctly")


if __name__ == "__main__":
    download_cub()