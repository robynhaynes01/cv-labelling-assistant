from util.bbox_manager import BBoxManager

class StandardFormat:
    def __init__(self):
        self._bboxes = []
        self._image_name = None
        self._image_width = None
        self._image_height = None
        self._source_format = None
        self._source_bbox_format = None
        self._target_format = 'yolo'
        self._bbox_manager = BBoxManager()

    def add_bbox(self, bbox, class_name):
        compiled_bbox = {
            'class': class_name,
            'bbox': bbox
        }
        self._bboxes.append(compiled_bbox)
    
    def set_image_name(self, image_name):
        self._image_name = image_name
    
    def set_image_dimensions(self, width, height):
        self._image_width = width
        self._image_height = height

    def set_source_format(self, source_format):
        self._source_format = source_format
    
    def set_source_bbox_format(self, source_bbox_format):
        self._source_bbox_format = source_bbox_format

    def set_excess_data(self, data: dict):
        self._excess_data = data

    def get_bboxes(self):
        return self._bboxes
    
    def normalize_bboxes(self):
        # Placeholder for bbox normalization logic
        for bbox in self._bboxes:
            normalized_box = self._bbox_manager.convert_bbox(bbox['bbox'], self._source_bbox_format, self._target_format, self._image_width, self._image_height)
            bbox['bbox'] = normalized_box

    def denoramize_bboxes(self):
        for bbox in self._bboxes:
            normalized_box = self._bbox_manager.convert_bbox(bbox['bbox'], self._target_format, self._source_bbox_format, self._image_width, self._image_height)
            bbox['bbox'] = normalized_box

    def __repr__(self):
        return f"StandardFormat(image_name={self._image_name}, image_width={self._image_width}, image_height={self._image_height}, source_format={self._source_format}, source_bbox_format={self._source_bbox_format}, target_format={self._target_format})"
