from ultralytics import settings

RUNS_DIR = settings.get('runs_dir')
DATASETS_DIR = 'datasets'

SUPPORTED_TASKS = ["detect"]

print(f"Runs: {RUNS_DIR}")
print(f"Datasets: {DATASETS_DIR}")