import json
class LabelHandlerInterface:
    def load_labels(self, file_path: str) -> list:
        raise NotImplementedError
    
    def save_labels(self, labels: list, file_path: str) -> None:
        raise NotImplementedError
    
    def convert_labels(self, labels: list, target_format: str) -> list:
        raise NotImplementedError
    
    def _determine_label_format(self, file_path: str) -> str:
        raise NotImplementedError
    
class LabelTXTHandler(LabelHandlerInterface):
    def load_labels(self, file_path: str) -> list:
        with open(file_path, 'r') as f:
            labels = f.readlines()
        return [label.strip() for label in labels]
    
    def save_labels(self, labels: list, file_path: str) -> None:
        with open(file_path, 'w') as f:
            for label in labels:
                f.write(f"{label}\n")

class LabelXMLHandler(LabelHandlerInterface):
    pass

class LabelJSONHandler(LabelHandlerInterface):
    def load_labels(self, file_path: str) -> list:
        with open(file_path, 'r') as f:
            labels = json.load(f)
        return labels
    
    def save_labels(self, labels: list, file_path: str) -> None:
        with open(file_path, 'w') as f:
            json.dump(labels, f, indent=4)


if __name__ == '__main__':
    label_handler = LabelJSONHandler()
    labels = label_handler.load_labels('path/to/labels.json')
    label_handler._determine_format('path/to/labels.json')