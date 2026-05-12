import os
import yaml
from loguru import logger as log
from dataset_manager.validator import validate_import_image_and_label_formats
from dataset_manager.common import VALID_IMAGE_FORMATS, VALID_LABEL_FORMATS, DATASET_CATEGORIES
from util.standard_format import StandardFormat
from util.data_source_manager import StandardSourceManager
from label_manager.manager import LabelManager

@log.catch
def populate_dataset(dataset_name, data_path, labels, classes):
    # This function will handle the actual copying of image and label files into the correct folder structure
    # and renaming them if necessary to ensure they match.
    log.debug(f"Populating dataset '{dataset_name}' at 'datasets/{dataset_name}'...")
    dataset_location = f'datasets/{dataset_name}'
    label_manager = LabelManager()
    
    for category in labels:
        category_images_folder = os.path.join(dataset_location, 'images', category)
        category_labels_folder = os.path.join(dataset_location, 'labels', category)

        for label in labels.get(category):
            image_name = label.get_image_name()
            image_source_path = os.path.join(data_path, image_name)
            image_destination_path = os.path.join(category_images_folder, image_name)
            log.info(f"Copying image from '{image_source_path}' to '{image_destination_path}'")
            if os.path.exists(image_source_path):
                os.rename(image_source_path, image_destination_path)
            else:
                raise ValueError(f"Image file '{image_name}' not found in source data path. Please ensure that all images referenced in the labels exist in the provided data path.")

            label_file_name = os.path.splitext(image_name)[0] + '.txt'  # Assuming YOLO format for labels
            label_destination_path = os.path.join(category_labels_folder, label_file_name)
            formatted_labels = []
            for bbox in label.get_bboxes():
                class_name = classes.index(bbox.get('class'))
                bbox_coordinates = bbox.get('bbox')
                formatted_labels.append(f"{class_name} {' '.join(map(str, bbox_coordinates))}")
            label_manager.save_labels(formatted_labels, label_destination_path)



def create_training_and_validation_split(labels: list[StandardFormat], split_ratio=0.8):
    log.info("Splitting labels into training and validation categories...")
    total_labels = len(labels)
    split_index = int(total_labels * split_ratio)
    log.debug(f"Split ratio is: {split_ratio}. This generates {split_index + 1} training labels and {(len(total_labels) - split_index) + 1} validation labels")
    training_labels = labels[:split_index]
    validation_labels = labels[split_index:]
    label_split = {'train': training_labels, 'val': validation_labels}
    return label_split

def generate_dataset_yaml(dataset_name, classes):
    log.info(f"Generating dataset yaml for '{dataset_name}', with classes: {', '.join(classes)}")
    yaml_data = {
        'train': 'images/train',
        'val': 'images/val',
        'names': classes
    }
    with open(f'datasets/{dataset_name}/{dataset_name}.yaml', 'w') as yaml_file:
        yaml.dump(yaml_data, yaml_file)

def convert_labels_to_standard_format(label_files, source_name, data_path) -> list[StandardFormat]:
    log.info(f"Converting labels from '{source_name}' into a standardised format...")
    standardised_labels = []
    manager = StandardSourceManager()
    for label_file in label_files:
        label_manager = LabelManager()
        raw_label_data = label_manager.load_labels(os.path.join(data_path, label_file))
        standardised_label_data = manager.normalize_source_data(source_name, raw_label_data)
        standardised_labels.append(standardised_label_data)
    return standardised_labels

def extract_unique_classes_from_labels(labels):
    log.info("Extracting all unique class names from provided labels...")
    unique_classes = set()
    for label in labels:
        for bbox in label.get_bboxes():
            unique_classes.add(bbox.get('class'))
    unique_classes = sorted(unique_classes)
    return unique_classes

def isolate_labels(data_path) -> list[str]:
    log.info(f"Isolating labels from '{data_path}'...")
    label_files = [file for file in os.listdir(data_path) if os.path.splitext(file)[1] in VALID_LABEL_FORMATS]
    return label_files

def isolate_images(data_path) -> list[str]:
    log.info(f"Isolating images from '{data_path}'...")
    image_files = [file for file in os.listdir(data_path) if os.path.splitext(file)[1] in VALID_IMAGE_FORMATS]
    return image_files

def generate_dataset_folder_structure(dataset_name):
    log.info(f"Generating the dataset file structure for '{dataset_name}'...")
    dataset_folder = f'datasets/{dataset_name}'
    if not os.path.exists(dataset_folder):
        log.debug("Generating root folder...")
        os.makedirs(dataset_folder)
    images_folder = os.path.join(dataset_folder, 'images')
    labels_folder = os.path.join(dataset_folder, 'labels')

    log.debug("Geneating image and label sub-directories...")
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
    if not os.path.exists(labels_folder):
        os.makedirs(labels_folder)

    log.debug(f"Generating image and label sub-categories: {', '.join(DATASET_CATEGORIES.keys())}")
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

    standardised_labels = convert_labels_to_standard_format(labels, source_name, data_path)

    classes = extract_unique_classes_from_labels(standardised_labels)

    # Generate the new yaml file for the dataset
    generate_dataset_yaml(dataset_name, classes)

    # Create the training and validation split
    label_split = create_training_and_validation_split(standardised_labels)

    populate_dataset(dataset_name, data_path, label_split, classes)

if __name__ == "__main__":
    create_new_dataset('test_dataset', 'labelme', 'test/mock_dataset')