from dataset_manager.validator import validate_import_image_and_label_formats

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