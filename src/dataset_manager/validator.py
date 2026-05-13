import os
from loguru import logger as log
from dataset_manager.common import DATASET_CATEGORIES, VALID_IMAGE_FORMATS, VALID_LABEL_FORMATS

@log.catch
def validate_image_format(image_file):
    if not os.path.splitext(image_file)[1].lower() in VALID_IMAGE_FORMATS:
        raise ValueError(f"Invalid image format for file '{image_file}'. Supported formats are: {', '.join(VALID_IMAGE_FORMATS)}.")

@log.catch
def validate_label_format(label_file):
    if not os.path.splitext(label_file)[1].lower() in VALID_LABEL_FORMATS:
        raise ValueError(f"Invalid label format for file '{label_file}'. Supported formats are: {', '.join(VALID_LABEL_FORMATS)}.")

@log.catch
def validate_import_image_and_label_formats(data_root):
    files = os.listdir(data_root)
    for file in files:
        if os.path.splitext(file)[1].lower() in VALID_IMAGE_FORMATS:
            validate_image_format(file)
        elif os.path.splitext(file)[1].lower() in VALID_LABEL_FORMATS:
            validate_label_format(file)
        else:
            raise ValueError(f"Unsupported file format for file '{file}' in dataset root directory. Supported image formats are: {', '.join(VALID_IMAGE_FORMATS)}. Supported label formats are: {', '.join(VALID_LABEL_FORMATS)}.")

@log.catch
def validate_image_and_label_names_match(dataset_root, training_category):
    images_dir = os.path.join(dataset_root, 'images', training_category)
    labels_dir = os.path.join(dataset_root, 'labels', training_category)

    image_files = sorted([os.path.splitext(name)[0] for name in os.listdir(images_dir)])
    label_files = sorted([os.path.splitext(name)[0] for name in os.listdir(labels_dir)])

    if image_files != label_files:
        raise ValueError(f"Image files and label files in '{training_category}' category do not match. A name mismatch has been detected, which may lead to issues during training. Please ensure that each image file has a corresponding label file with the same name (excluding the file extension).")

@log.catch
def validate_training_categories(dataset_root):
    images_dir = os.path.join(dataset_root, 'images')
    labels_dir = os.path.join(dataset_root, 'labels')

    image_categories = sorted(os.listdir(images_dir))
    label_categories = sorted(os.listdir(labels_dir))

    for category in image_categories:
        if category not in DATASET_CATEGORIES.keys():
            raise ValueError(f"Invalid training category '{category}' found in 'images' subdirectory. Please ensure that the 'images' subdirectory contains only valid training categories: {', '.join(list(DATASET_CATEGORIES.keys()))}.")
    for category in label_categories:
        if category not in DATASET_CATEGORIES.keys():
            raise ValueError(f"Invalid training category '{category}' found in 'labels' subdirectory. Please ensure that the 'labels' subdirectory contains only valid training categories: {', '.join(list(DATASET_CATEGORIES.keys()))}.")

    for category in DATASET_CATEGORIES.keys():
        optional_category = DATASET_CATEGORIES.get(category).get('optional', False)
        if category not in image_categories and not optional_category:
            raise ValueError(f"Mandatory training category '{category}' is missing in 'images' subdirectory. Please ensure that the 'images' subdirectory contains the mandatory training category: '{category}'.")
        if category not in label_categories and not optional_category:
            raise ValueError(f"Mandatory training category '{category}' is missing in 'labels' subdirectory. Please ensure that the 'labels' subdirectory contains the mandatory training category: '{category}'.")

    if image_categories != label_categories:
        raise ValueError(f"Training categories in 'images' and 'labels' subdirectories do not match. A category mismatch has been detected, which may lead to issues during training. Please ensure that the same set of training categories (e.g., 'train', 'val', 'test') exist in both 'images' and 'labels' subdirectories.") 

@log.catch
def validate_images_and_labels(dataset_root):
    images = os.path.join(dataset_root, 'images')
    labels = os.path.join(dataset_root, 'labels')

    if len(os.listdir(images)) == 0 or len(os.listdir(labels)) == 0:
        raise ValueError(f"Dataset root directory '{dataset_root}' must contain non-empty 'images' and 'labels' subdirectories.")
    if sorted(os.listdir(images)) != sorted(os.listdir(labels)):
        raise ValueError(f"Dataset root directory '{dataset_root}' must contain matching subdirectories in 'images' and 'labels' subdirectories.")

@log.catch
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
