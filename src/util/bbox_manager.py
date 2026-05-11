from util.bbox_handlers import YOLOHandler, PascalVOCHandler, COCOHandler
from util.common import YOLO, PASCAL_VOC, COCO

class BBoxManager:
    def __init__(self):
        self._transformers = {
            YOLO: YOLOHandler(),
            PASCAL_VOC: PascalVOCHandler(),
            COCO: COCOHandler()
        }

    def convert_bbox(self, bbox, source_format, target_format, image_width=None, image_height=None):
        if source_format not in self._transformers:
            raise ValueError(f"Unsupported source format: {source_format}")
        if target_format not in self._transformers:
            raise ValueError(f"Unsupported target format: {target_format}")
        
        transformer = self._transformers[source_format]
        if target_format == YOLO:
            return transformer.to_cxcywh(bbox, image_width, image_height)
        elif target_format == PASCAL_VOC:
            return transformer.to_xyxy(bbox, image_width, image_height)
        elif target_format == COCO:
            return transformer.to_xywh(bbox)
        
