import os
import pathlib

def validate_image_and_label_names_match(dataset_root, training_category):
    images_dir = os.path.join(dataset_root, 'images', training_category)
    labels_dir = os.path.join(dataset_root, 'labels', training_category)

    image_files = sorted([os.path.splitext(name)[0] for name in os.listdir(images_dir)])
    label_files = sorted([os.path.splitext(name)[0] for name in os.listdir(labels_dir)])

    if image_files != label_files:
        raise ValueError(f"Image files and label files in '{training_category}' category do not match. A name mismatch has been detected, which may lead to issues during training. Please ensure that each image file has a corresponding label file with the same name (excluding the file extension).")

def validate_training_categories(dataset_root):
    images_dir = os.path.join(dataset_root, 'images')
    labels_dir = os.path.join(dataset_root, 'labels')

    image_categories = sorted(os.listdir(images_dir))
    label_categories = sorted(os.listdir(labels_dir))

    categories = {'train': {'cat_name': 'train', 'optional': False}, 'val': {'cat_name': 'val', 'optional': False}, 'test': {'cat_name': 'test', 'optional': True}}

    for category in image_categories:
        if category not in categories.keys():
            raise ValueError(f"Invalid training category '{category}' found in 'images' subdirectory. Please ensure that the 'images' subdirectory contains only valid training categories: {', '.join(list(categories.keys()))}.")
    for category in label_categories:
        if category not in categories.keys():
            raise ValueError(f"Invalid training category '{category}' found in 'labels' subdirectory. Please ensure that the 'labels' subdirectory contains only valid training categories: {', '.join(list(categories.keys()))}.")

    for category in categories.keys():
        valid_category = categories.get(category)
        if valid_category.get('cat_name') not in image_categories and not valid_category.get('optional'):
            raise ValueError(f"Mandatory training category '{category}' is missing in 'images' subdirectory. Please ensure that the 'images' subdirectory contains the mandatory training category: '{category}'.")
        if valid_category.get('cat_name') not in label_categories and not valid_category.get('optional'):
            raise ValueError(f"Mandatory training category '{category}' is missing in 'labels' subdirectory. Please ensure that the 'labels' subdirectory contains the mandatory training category: '{category}'.")

    if image_categories != label_categories:
        raise ValueError(f"Training categories in 'images' and 'labels' subdirectories do not match. A category mismatch has been detected, which may lead to issues during training. Please ensure that the same set of training categories (e.g., 'train', 'val', 'test') exist in both 'images' and 'labels' subdirectories.") 


def validate_images_and_labels(dataset_root):
    images = os.path.join(dataset_root, 'images')
    labels = os.path.join(dataset_root, 'labels')

    if len(os.listdir(images)) == 0 or len(os.listdir(labels)) == 0:
        raise ValueError(f"Dataset root directory '{dataset_root}' must contain non-empty 'images' and 'labels' subdirectories.")
    if sorted(os.listdir(images)) != sorted(os.listdir(labels)):
        raise ValueError(f"Dataset root directory '{dataset_root}' must contain matching subdirectories in 'images' and 'labels' subdirectories.")

def validate_root_dataset_directory(dataset_root):
    if not os.path.exists(dataset_root):
        raise ValueError(f"Dataset root directory '{dataset_root}' does not exist.")
    if not os.path.isdir(dataset_root):
        raise ValueError(f"Dataset root path '{dataset_root}' is not a directory.")
    
    children = os.listdir(dataset_root)
    if len(children) == 0:
        raise ValueError(f"Dataset root directory '{dataset_root}' is empty.")
    
    if not all(x in children for x in ['images', 'labels']):
        raise ValueError(f"Dataset root directory '{dataset_root}' must contain 'images' and 'labels' subdirectories.")

    dataset_name = os.path.basename(dataset_root)
    if f"{dataset_name}.yaml" not in children and f"{dataset_name}.yml" not in children:
        raise ValueError(f"Dataset root directory '{dataset_root}' must contain a YAML config file named '{dataset_name}.yaml' or '{dataset_name}.yml'.")

def validate_dataset_format(dataset_path):
    # Validation logic for dataset format, mainly for YOLO models
    # Example of the expected format:
    # datasets/
    # ├── my_custom_dataset/
    # │   ├── images/
    # |   |   ├── train/
    # |   │   │   ├── img1.jpg
    # |   │   │   ├── img2.jpg
    # |   │   │   └── ...
    # |   |   ├── val/
    # |   │   │   ├── img1.jpg
    # |   │   │   ├── img2.jpg
    # |   │   │   └── ...
    # |   |   ├── test/ (optional)
    # |   │   │   ├── img1.jpg
    # |   │   │   ├── img2.jpg
    # |   │   │   └── ...
    # │   └── labels/
    # |   |   ├── train/
    # |   │   │   ├── img1.jpg
    # |   │   │   ├── img2.jpg
    # |   │   │   └── ...
    # |   |   ├── val/
    # |   │   │   ├── img1.txt
    # |   │   │   ├── img2.txt
    # |   │   │   └── ...
    # |   |   ├── test/ (optional)
    # |   │   │   ├── img1.jpg
    # |   │   │   ├── img2.jpg
    # |   │   │   └── ...
    # |   └── my_custom_dataset.yaml
    # └── ... other datasets ...
    validate_root_dataset_directory(dataset_path)
    validate_images_and_labels(dataset_path)
    validate_training_categories(dataset_path)
    categories = os.listdir(os.path.join(dataset_path, 'images'))
    for category in categories:
        validate_image_and_label_names_match(dataset_path, category)
    

if __name__ == "__main__":
    dataset_path = "datasets/coco8"
    try:
        if validate_dataset_format(dataset_path):
            print(f"Dataset format for '{dataset_path}' is valid.")
    except ValueError as e:
        print(f"Dataset format validation error: {e}")