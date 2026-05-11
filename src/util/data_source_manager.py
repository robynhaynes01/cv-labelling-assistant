from util.standard_format import StandardFormat
from util.common import PASCAL_VOC, LABELME

class SourceManagerInterface:
    def convert_to_standard_format(self, data):
        raise NotImplementedError
    
    def convert_from_standard_format(self, data):
        raise NotImplementedError
    
    def normalize_source_bbox(self):
        raise NotImplementedError
    
    def denormalize_source_bbox(self):
        raise NotImplementedError

class LabelMeSourceManager(SourceManagerInterface):
    def convert_to_standard_format(self, data):
        shapes = data.get('shapes', [])
        
        standard_format = StandardFormat()
        image_width = data.get('imageWidth')
        image_height = data.get('imageHeight')

        for shape in shapes:
            label = shape.get('label')
            points = shape.get('points')
            image_name = data.get('imagePath')
            standard_format.add_bbox(self.normalize_source_bbox(points), label)

        standard_format.set_image_name(image_name)
        standard_format.set_image_dimensions(image_width, image_height)
        standard_format.set_source_format(LABELME)
        standard_format.set_source_bbox_format(PASCAL_VOC)
        standard_format.normalize_bboxes()

        return standard_format

    def convert_from_standard_format(self, data: StandardFormat):
        data.denoramize_bboxes()
        return {
            'version': '',
            'flags': {},
            'shapes': [{'label': bbox['class'], 'points': self.denormalize_source_bbox(bbox['bbox']), 'group_id': None, 'shape_type': 'rectangle', 'flags': {}} for bbox in data._bboxes],
            'imagePath': data._image_name,
            'imageData': None,
            'imageHeight': data._image_height,
            'imageWidth': data._image_width
        }


    def normalize_source_bbox(self, box_data):
        # For LabelMe, we flatten the list of points into a single dimension
        flattened_box = [coord for point in box_data for coord in point]
        return flattened_box
    
    def denormalize_source_bbox(self, bbox):
        # Convert from Pascal VOC format to LabelMe format
        x_min, y_min, x_max, y_max = bbox
        return [[x_min, y_min], [x_max, y_max]]
    
class StandardSourceManager():
    def __init__(self):
        self._source_mangers = {
            LABELME: LabelMeSourceManager(),
        }

    def normalize_source_data(self, source_name, data):
        manager = self._source_mangers.get(source_name)
        if not manager:
            raise ValueError(f"Unsupported source: {source_name}")
        return manager.convert_to_standard_format(data)
    
    def denormalize_to_source_data(self, source_name, data: StandardFormat):
        manager = self._source_mangers.get(source_name)
        if not manager:
            raise ValueError(f"Unsupported source: {source_name}")
        return manager.convert_from_standard_format(data)
    
if __name__ == "__main__":
    import json

    with open('test/utils/test_data.json', 'r') as f:
        test_data = json.load(f)

    manager = StandardSourceManager()
    standard_format = manager.normalize_source_data('labelme', test_data)
    print(standard_format.get_bboxes())
    labelme_data = manager.denormalize_to_source_data('labelme', standard_format)
    print(labelme_data)