# This file holds as a common place for constants used throughout the dataset management code
# Adding here with future constants will help ensure that single changes or new support
# Is easily reflected across the module.

### IMAGE AND LABEL FORMATS ###
VALID_IMAGE_FORMATS = ('.jpg', '.jpeg', '.png')
VALID_LABEL_FORMATS = ('.txt', '.json', '.xml')

### DATASET CATEGORIES ###
DATASET_CATEGORIES = {
    'train': {
        'optional': False
    },
    'val': {
        'optional': False
    },
    'test': {
        'optional': True
    }
}