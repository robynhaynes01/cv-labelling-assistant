import os
import json
from dataset_manager.creator import create_new_dataset
from dataset_manager.update import update_existing_dataset

def generate_test_dataset(fs, dataset_root):
    # This function is used to generate a test dataset for testing the dataset creation and update processes.
    # It creates a folder structure with some mock images and labels in the 'test/mock_dataset' directory.
    if not os.path.exists(dataset_root):
        os.makedirs(dataset_root)

    # Create some mock images (empty files for testing)
    image_filenames = ['image1.jpg', 'image2.jpg']
    for image_filename in image_filenames:
        image_path = os.path.join(dataset_root, image_filename)
        with open(image_path, 'w') as f:
            f.write('')  # Create an empty file

    # Create some mock label files in LabelMe format
    label_data_1 = {
        "version": "4.5.6",
        "flags": {},
        "shapes": [
            {
                "label": "cat",
                "points": [[10, 10], [100, 100]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            },
            {
                "label": "vase",
                "points": [[150, 150], [200, 200]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            }
        ],
        "imagePath": "image1.jpg",
        "imageData": None,
        "imageHeight": 500,
        "imageWidth": 500
    }

    label_data_2 = {
        "version": "4.5.6",
        "flags": {},
        "shapes": [
            {
                "label": "dog",
                "points": [[20, 20], [120, 120]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            },
            {
                "label": "elephant",
                "points": [[250, 250], [300, 300]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            }
        ],
        "imagePath": "image2.jpg",
        "imageData": None,
        "imageHeight": 500,
        "imageWidth": 500
    }
    labels = [label_data_1, label_data_2]
    for label in labels:
        label_filename = f"{os.path.splitext(label['imagePath'])[0]}.json"
        label_path = os.path.join(dataset_root, label_filename)
        with open(label_path, 'w') as f:
            json.dump(label, f, indent=4)

def generate_test_dataset_additional(fs, dataset_root):
    # This function is used to generate a test dataset for testing the dataset creation and update processes.
    # It creates a folder structure with some mock images and labels in the 'test/mock_dataset' directory.
    if not os.path.exists(dataset_root):
        os.makedirs(dataset_root)

    # Create some mock images (empty files for testing)
    image_filenames = ['image3.jpg', 'image4.jpg']
    for image_filename in image_filenames:
        image_path = os.path.join(dataset_root, image_filename)
        with open(image_path, 'w') as f:
            f.write('')  # Create an empty file

    # Create some mock label files in LabelMe format
    label_data_1 = {
        "version": "4.5.6",
        "flags": {},
        "shapes": [
            {
                "label": "cat",
                "points": [[10, 10], [100, 100]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            },
            {
                "label": "vase",
                "points": [[150, 150], [200, 200]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            }
        ],
        "imagePath": "image3.jpg",
        "imageData": None,
        "imageHeight": 500,
        "imageWidth": 500
    }

    label_data_2 = {
        "version": "4.5.6",
        "flags": {},
        "shapes": [
            {
                "label": "dog",
                "points": [[20, 20], [120, 120]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            },
            {
                "label": "elephant",
                "points": [[250, 250], [300, 300]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            }
        ],
        "imagePath": "image4.jpg",
        "imageData": None,
        "imageHeight": 500,
        "imageWidth": 500
    }
    labels = [label_data_1, label_data_2]
    for label in labels:
        label_filename = f"{os.path.splitext(label['imagePath'])[0]}.json"
        label_path = os.path.join(dataset_root, label_filename)
        with open(label_path, 'w') as f:
            json.dump(label, f, indent=4)

def test_dataset_creation(fs):
    dataset_root = 'test/mock_dataset'
    generate_test_dataset(fs, dataset_root)

    create_new_dataset('test_dataset', 'labelme', dataset_root)
    assert os.path.exists('datasets/test_dataset/test_dataset.yaml'), "YAML config file was not created."
    assert len(os.listdir('datasets/test_dataset/images/train')) > 0, "Images were not copied to the training folder."
    assert len(os.listdir('datasets/test_dataset/labels/train')) > 0, "Labels were not copied to the training folder."
    assert len(os.listdir('datasets/test_dataset/images/val')) > 0, "Images were not copied to the validation folder."
    assert len(os.listdir('datasets/test_dataset/labels/val')) > 0, "Labels were not copied to the validation folder."

def test_dataset_update(fs):
    dataset_root = 'test/mock_dataset'
    if not os.path.exists(dataset_root):
        generate_test_dataset(fs, dataset_root)
        create_new_dataset('test_dataset', 'labelme', dataset_root)

    os.remove('test/mock_dataset/image1.json')
    os.remove('test/mock_dataset/image2.json')
    generate_test_dataset_additional(fs, dataset_root)
    update_existing_dataset('test_dataset', 'labelme', dataset_root)

    assert os.path.exists('datasets/test_dataset/test_dataset.yaml'), "YAML config file was not created."
    assert len(os.listdir('datasets/test_dataset/images/train')) > 1, "Images were not copied to the training folder."
    assert len(os.listdir('datasets/test_dataset/labels/train')) > 1, "Labels were not copied to the training folder."
    assert len(os.listdir('datasets/test_dataset/images/val')) > 1, "Images were not copied to the validation folder."
    assert len(os.listdir('datasets/test_dataset/labels/val')) > 1, "Labels were not copied to the validation folder."
