import os
import pyfakefs

from dataset_manager.validator import validate_dataset_format

def create_fake_dataset_structure(fs, dataset_root):
    fs.create_dir(os.path.join(dataset_root, 'images', 'train'))
    fs.create_dir(os.path.join(dataset_root, 'labels', 'train'))
    fs.create_dir(os.path.join(dataset_root, 'images', 'val'))
    fs.create_dir(os.path.join(dataset_root, 'labels', 'val'))
    fs.create_file(os.path.join(dataset_root, 'images', 'train', 'img1.jpg'))
    fs.create_file(os.path.join(dataset_root, 'labels', 'train', 'img1.txt'))
    fs.create_file(os.path.join(dataset_root, 'images', 'val', 'img1.jpg'))
    fs.create_file(os.path.join(dataset_root, 'labels', 'val', 'img1.txt'))

def create_fake_dataset_structure_invalid_category(fs, dataset_root):
    fs.create_dir(os.path.join(dataset_root, 'images', 'train'))
    fs.create_dir(os.path.join(dataset_root, 'labels', 'train'))
    fs.create_dir(os.path.join(dataset_root, 'images', 'val'))
    fs.create_dir(os.path.join(dataset_root, 'labels', 'val'))
    fs.create_dir(os.path.join(dataset_root, 'images', 'oops'))
    fs.create_dir(os.path.join(dataset_root, 'labels', 'oops'))
    fs.create_file(os.path.join(dataset_root, 'images', 'train', 'img1.jpg'))
    fs.create_file(os.path.join(dataset_root, 'labels', 'train', 'img1.txt'))
    fs.create_file(os.path.join(dataset_root, 'images', 'val', 'img1.jpg'))
    fs.create_file(os.path.join(dataset_root, 'labels', 'val', 'img1.txt'))


def create_fake_yaml_config(fs, dataset_root):
    dataset_name = os.path.basename(dataset_root)
    fs.create_file(os.path.join(dataset_root, f"{dataset_name}.yaml"))

def delete_fake_yaml_config(fs, dataset_root):
    dataset_name = os.path.basename(dataset_root)
    yaml_path = os.path.join(dataset_root, f"{dataset_name}.yaml")
    if os.path.exists(yaml_path):
        fs.remove_object(yaml_path)

# Test case to error ValueError when YAML config file is missing
def test_yaml_config_missing(fs):
    dataset_root = '/fake_dataset'
    create_fake_dataset_structure(fs, dataset_root)

    try:
        validate_dataset_format(dataset_root)
        assert False, "Expected ValueError for missing YAML config file"
    except ValueError as e:
        assert str(e) == f"Dataset root directory '{dataset_root}' must contain a YAML config file named '{os.path.basename(dataset_root)}.yaml' or '{os.path.basename(dataset_root)}.yml'."


def test_validate_images_and_labels_valid_structure(fs):
    dataset_root = '/fake_dataset'
    create_fake_dataset_structure(fs, dataset_root)
    create_fake_yaml_config(fs, dataset_root)


    validate_dataset_format(dataset_root)

def test_validate_images_and_labels_invalid_category(fs):
    dataset_root = '/fake_dataset'
    create_fake_dataset_structure_invalid_category(fs, dataset_root)
    create_fake_yaml_config(fs, dataset_root)

    try:
        validate_dataset_format(dataset_root)
        assert False, "Expected ValueError for invalid training category in 'images' subdirectory"
    except ValueError as e:
        assert str(e) == f"Invalid training category 'oops' found in 'images' subdirectory. Please ensure that the 'images' subdirectory contains only valid training categories: train, val, test."
