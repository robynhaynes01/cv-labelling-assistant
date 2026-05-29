from ultralytics.cfg import TASK2MODEL
from ultralytics import YOLO
from model_manager.common import DETECT, DATASETS_DIR

class ModelTaskInterface():
    def get_dataset_path(self, dataset_name):
        dataset_path = f"{DATASETS_DIR}/{dataset_name}/{dataset_name}.yaml"
        return dataset_path

    def train_model(self, dataset_name: str, device: list, epochs: int):
        """Interface for different task handlers to share for a unified training experience"""
        raise NotImplementedError
    
class ModelDetectionTask(ModelTaskInterface):
    def train_model(self, dataset_name, device, epochs):
        model_name = TASK2MODEL.get(DETECT)
        dataset_path = self.get_dataset_path(dataset_name)
        model = YOLO(model=model_name, task=DETECT)
        results = model.train(data=dataset_path, project=dataset_name, device=device, epochs=epochs)

if __name__ == '__main__':
    mdt = ModelDetectionTask()
    mdt.train_model("test_dataset", "cuda", 10)