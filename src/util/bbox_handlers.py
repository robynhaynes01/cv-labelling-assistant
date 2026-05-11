class BBoxHandlerInterface:
    def _get_dw_dh(self, image_width, image_height):
        if image_width is None or image_height is None:
            dw = 1
            dh = 1
        else:
            dw = 1. / image_width
            dh = 1. / image_height
        return dw, dh

    def to_xyxy(self, bbox, image_width=None, image_height=None):
        """Convert bounding box to (x_min, y_min, x_max, y_max) format. Aka Pascal VOC format."""
        raise NotImplementedError
    
    def to_cxcywh(self, bbox, image_width=None, image_height=None):
        """Convert bounding box to (center_x, center_y, width, height) format. Aka YOLO format."""
        raise NotImplementedError
    
    def to_xywh(self, bbox, image_width=None, image_height=None):
        """Convert bounding box to (x_min, y_min, width, height) format. Aka COCO format."""
        raise NotImplementedError


class YOLOHandler(BBoxHandlerInterface):
    def to_xyxy(self, bbox, image_width, image_height):
        # Convert from (center_x, center_y, width, height) to (x_min, y_min, x_max, y_max)
        center_x, center_y, width, height = bbox
        x_min = (center_x - width / 2) * image_width
        y_min = (center_y - height / 2) * image_height
        x_max = (center_x + width / 2) * image_width
        y_max = (center_y + height / 2) * image_height
        return (x_min, y_min, x_max, y_max)
    
    def to_cxcywh(self, bbox):
        return bbox
    
    def to_xywh(self, bbox):
        # Convert from (center_x, center_y, width, height) to (x_min, y_min, width, height)
        center_x, center_y, width, height = bbox
        x_min = center_x - width / 2
        y_min = center_y - height / 2
        return (x_min, y_min, width, height)
    
class PascalVOCHandler(BBoxHandlerInterface):
    def to_xyxy(self, bbox):
        return bbox
    
    def to_cxcywh(self, bbox, image_width, image_height):
        # Convert from (x_min, y_min, x_max, y_max) to (center_x, center_y, width, height)
        x_min, y_min, x_max, y_max = bbox
        dw, dh = self._get_dw_dh(image_width, image_height)

        center_x = (x_min + x_max) / 2 * dw
        center_y = (y_min + y_max) / 2 * dh
        width = (x_max - x_min) * dw
        height = (y_max - y_min) * dh
        return (center_x, center_y, width, height)
    
    def to_xywh(self, bbox):
        # Convert from (x_min, y_min, x_max, y_max) to (x_min, y_min, width, height)
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        return (x_min, y_min, width, height)
    
class COCOHandler(BBoxHandlerInterface):
    def to_xyxy(self, bbox):
        # Convert from (x_min, y_min, width, height) to (x_min, y_min, x_max, y_max)
        x_min, y_min, width, height = bbox
        x_max = x_min + width
        y_max = y_min + height
        return (x_min, y_min, x_max, y_max)
    
    def to_cxcywh(self, bbox):
        # Convert from (x_min, y_min, width, height) to (center_x, center_y, width, height)
        x_min, y_min, width, height = bbox
        center_x = x_min + width / 2
        center_y = y_min + height / 2
        return (center_x, center_y, width, height)
    
    def to_xywh(self, bbox):
        return bbox