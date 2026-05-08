import os
from dataset_manager.validator import validate_import_image_and_label_formats
from dataset_manager.common import VALID_IMAGE_FORMATS, VALID_LABEL_FORMATS
from util.data_source_manager import StandardSourceManager
from label_manager.manager import LabelManager

def isolate_labels(data_path) -> list[str]:
    label_files = [file for file in os.listdir(data_path) if os.path.splitext(file)[1] in VALID_LABEL_FORMATS]
    return label_files

def isolate_images(data_path) -> list[str]:
    image_files = [file for file in os.listdir(data_path) if os.path.splitext(file)[1] in VALID_IMAGE_FORMATS]
    return image_files


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

if __name__ == "__main__":
    create_new_dataset('test_dataset', 'labelme', 'test/mock_dataset')