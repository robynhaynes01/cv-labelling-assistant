import os
import yaml
from dataset_manager.creator import convert_labels_to_standard_format, isolate_labels, extract_unique_classes_from_labels, create_training_and_validation_split, generate_dataset_yaml, populate_dataset

def merge_classes(existing_classes, new_classes):
    unique_new_classes = set(new_classes) - set(existing_classes)
    for class_name in unique_new_classes:
        existing_classes.append(class_name)

    return existing_classes

def load_existing_classes(dataset_name):
    with open(f"datasets/{dataset_name}/{dataset_name}.yaml", "r") as f:
        dataset_info = yaml.safe_load(f)
    return dataset_info.get("names", [])

def update_existing_dataset(dataset_name, source_name, data_path):
    # Check if dataset exists
    if not os.path.exists(f"datasets/{dataset_name}"):
        print(f"Dataset '{dataset_name}' does not exist.")
        return
    
    labels = isolate_labels(data_path)
    
    standardised_labels = convert_labels_to_standard_format(labels, source_name, data_path)

    # Load existing dataset classes
    existing_classes = load_existing_classes(dataset_name)
    new_unique_classes = extract_unique_classes_from_labels(standardised_labels)
    print(f"Existing classes: {existing_classes}")
    print(f"New unique classes from imported labels: {new_unique_classes}")

    combined_classes = merge_classes(existing_classes, new_unique_classes)
    print(f"Combined classes: {combined_classes}")

    generate_dataset_yaml(dataset_name, combined_classes)

    label_split = create_training_and_validation_split(standardised_labels)

    populate_dataset(dataset_name, data_path, label_split, combined_classes)

if __name__ == "__main__":
    dataset_name = "test_dataset"
    data_path = "test/mock_dataset"
    update_existing_dataset(dataset_name, 'labelme', data_path)