import os
from label_manager.handlers import LabelTXTHandler, LabelXMLHandler, LabelJSONHandler

class LabelManager:
    def __init__(self):
        self.labels = {}
        self._handlers = {
            '.txt': LabelTXTHandler(),
            '.xml': LabelXMLHandler(),
            '.json': LabelJSONHandler()
        }

    def load_labels(self, file_path: str) -> list:
        file_extension = self._determine_label_format(file_path)
        handler = self._handlers.get(file_extension)
        if not handler:
            raise ValueError(f"Unsupported label format: {file_extension}")
        return handler.load_labels(file_path)
        
    def _determine_label_format(self, file_path: str) -> str:
        return os.path.splitext(file_path)[1]

    
if __name__ == '__main__':
    label_manager = LabelManager()
    labels = label_manager.load_labels('path/to/labels.json')
    print(labels)