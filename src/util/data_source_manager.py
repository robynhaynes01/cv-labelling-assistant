class SourceManagerInterface:
    def normalize_source(self):
        raise NotImplementedError
    
    def denormalize_source(self):
        raise NotImplementedError
    
class LabelMeSourceManager(SourceManagerInterface):
    def normalize_source(self, box_data):
        # For LabelMe, we can directly use the Pascal VOC format
        flattened_box = [coord for point in box_data for coord in point]
        return flattened_box, 'pascal_voc'
    
    def denormalize_source(self, bbox):
        # Convert from Pascal VOC format to LabelMe format
        x_min, y_min, x_max, y_max = bbox
        return [[x_min, y_min], [x_max, y_max]]
    