from ultralytics import settings

RUNS_DIR = settings.get('runs_dir')
DATASETS_DIR = 'datasets'

DETECT = "detect"
SUPPORTED_TASKS = [DETECT]
