import os
import yaml
from dataset_manager.validator import validate_import_image_and_label_formats
from dataset_manager.common import VALID_IMAGE_FORMATS, VALID_LABEL_FORMATS, DATASET_CATEGORIES
from util.data_source_manager import StandardSourceManager
from label_manager.manager import LabelManager

def generate_dataset_yaml(dataset_name, classes):
    yaml_data = {
        'train': 'images/train',
        'val': 'images/val',
        'names': classes
    }
    with open(f'datasets/{dataset_name}/{dataset_name}.yaml', 'w') as yaml_file:
        yaml.dump(yaml_data, yaml_file)

def extract_unique_classes_from_labels(labels):
    unique_classes = set()
    for label in labels:
        for bbox in label.get_bboxes():
            unique_classes.add(bbox.get('class'))
    unique_classes = sorted(unique_classes)
    return unique_classes

def isolate_labels(data_path) -> list[str]:
    label_files = [file for file in os.listdir(data_path) if os.path.splitext(file)[1] in VALID_LABEL_FORMATS]
    return label_files

def isolate_images(data_path) -> list[str]:
    image_files = [file for file in os.listdir(data_path) if os.path.splitext(file)[1] in VALID_IMAGE_FORMATS]
    return image_files

def generate_dataset_folder_structure(dataset_name):
    dataset_folder = f'datasets/{dataset_name}'
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    images_folder = os.path.join(dataset_folder, 'images')
    labels_folder = os.path.join(dataset_folder, 'labels')

    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
    if not os.path.exists(labels_folder):
        os.makedirs(labels_folder)

    training_categories = DATASET_CATEGORIES.keys()
    for category in training_categories:
        category_images_folder = os.path.join(images_folder, category)
        category_labels_folder = os.path.join(labels_folder, category)

        if not os.path.exists(category_images_folder):
            os.makedirs(category_images_folder)
        if not os.path.exists(category_labels_folder):
            os.makedirs(category_labels_folder)

def create_new_dataset(dataset_name, source_name, data_path):
    # Initially, dataset creation is relying on a folder structure
    # that is flat. In the future, this can be expended to support more
    # dynamic structures.
    # For now, we will create a folder structure like:
    # dataset_name/
    #   img1.jpg
    #   img1.txt
    #   img2.jpg
    #   img2.txt
    #   ...

    # Ensure the images and labels are in the correct format and can be imported
    validate_import_image_and_label_formats(data_path)

    # Create the dataset folder, provided it doesn't exist already
    generate_dataset_folder_structure(dataset_name)

    # Extract the labels and images from the provided data path
    labels = isolate_labels(data_path)
    images = isolate_images(data_path)

    standardised_labels = []
    manager = StandardSourceManager()
    for label_file in labels:
        label_manager = LabelManager()
        raw_label_data = label_manager.load_labels(os.path.join(data_path, label_file))
        standardised_label_data = manager.normalize_source_data(source_name, raw_label_data)
        standardised_labels.append(standardised_label_data)

    classes = extract_unique_classes_from_labels(standardised_labels)

    # Generate the new yaml file for the dataset
    generate_dataset_yaml(dataset_name, classes)

if __name__ == "__main__":
    create_new_dataset('test_dataset', 'labelme', 'test/mock_dataset')