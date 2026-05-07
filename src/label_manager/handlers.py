class LabelHandlerInterface:
    def load_labels(self, file_path: str) -> list:
        raise NotImplementedError
    
    def save_labels(self, labels: list, file_path: str) -> None:
        raise NotImplementedError
    
    def convert_labels(self, labels: list, target_format: str) -> list:
        raise NotImplementedError
    
    def _determine_format(self, file_path: str) -> str:
        raise NotImplementedError
    

class LabelTXTHandler(LabelHandlerInterface):
    pass

class LabelXMLHandler(LabelHandlerInterface):
    pass

class LabelJSONHandler(LabelHandlerInterface):
    pass


if __name__ == '__main__':
    label_handler = LabelJSONHandler()
    labels = label_handler.load_labels('path/to/labels.json')
    label_handler._determine_format('path/to/labels.json')