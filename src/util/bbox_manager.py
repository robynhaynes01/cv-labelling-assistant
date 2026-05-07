from util.bbox_handlers import YOLOHandler, PascalVOCHandler, COCOHandler

class BBoxManager:
    def __init__(self):
        self._transformers = {
            'yolo': YOLOHandler(),
            'pascal_voc': PascalVOCHandler(),
            'coco': COCOHandler()
        }

    def convert_bbox(self, bbox, source_format, target_format):
        if source_format not in self._transformers:
            raise ValueError(f"Unsupported source format: {source_format}")
        if target_format not in self._transformers:
            raise ValueError(f"Unsupported target format: {target_format}")
        
        transformer = self._transformers[source_format]
        if target_format == 'yolo':
            return transformer.to_cxcywh(bbox)
        elif target_format == 'pascal_voc':
            return transformer.to_xyxy(bbox)
        elif target_format == 'coco':
            return transformer.to_xywh(bbox)
        
