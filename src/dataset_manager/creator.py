import os
from dataset_manager.validator import validate_import_image_and_label_formats
from dataset_manager.common import VALID_IMAGE_FORMATS, VALID_LABEL_FORMATS

def isolate_labels(data_path) -> list[str]:
    label_files = [file for file in os.listdir(data_path) if os.path.splitext(file)[1] in VALID_LABEL_FORMATS]
    return label_files

def isolate_images(data_path) -> list[str]:
    image_files = [file for file in os.listdir(data_path) if os.path.splitext(file)[1] in VALID_IMAGE_FORMATS]
    return image_files


def create_new_dataset(dataset_name, data_path):
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
    validate_import_image_and_label_formats(data_path)
    pass