import os
import yaml

def load_existing_classes(dataset_name):
    with open(f"datasets/{dataset_name}/{dataset_name}.yaml", "r") as f:
        dataset_info = yaml.safe_load(f)
    return dataset_info.get("names", [])

def update_existing_dataset(dataset_name, data_path):
    # Check if dataset exists
    if not os.path.exists(f"datasets/{dataset_name}"):
        print(f"Dataset '{dataset_name}' does not exist.")
        return
    
    # Load existing dataset classes
    existing_classes = load_existing_classes(dataset_name)
    print(f"Existing classes in '{dataset_name}': {existing_classes}")

if __name__ == "__main__":
    dataset_name = "test_dataset"
    data_path = "path/to/new/data"
    update_existing_dataset(dataset_name, data_path)