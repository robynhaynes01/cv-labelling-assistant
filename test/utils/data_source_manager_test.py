import pytest

def test_source_manager_conversion_to_standard_format():
    from util.data_source_manager import StandardSourceManager
    import json

    with open('test/utils/test_data.json', 'r') as f:
        test_data = json.load(f)

    manager = StandardSourceManager()
    standard_format = manager.normalize_source_data('labelme', test_data)

    assert standard_format._image_name == "000000000009.jpg"
    assert standard_format._image_width == 640
    assert standard_format._image_height == 480
    assert standard_format._source_format == 'labelme'
    assert standard_format._source_bbox_format == 'pascal_voc'
    assert len(standard_format._bboxes) == 1
    assert standard_format._bboxes[0].get('class') == "test"
    assert standard_format._bboxes[0].get('bbox') == pytest.approx([0.31352486559139786, 0.6691868279569892, 0.4377352150537635, 0.9216173835125447])

def test_source_manager_conversion_from_standard_format():
    from util.data_source_manager import StandardSourceManager
    from util.standard_format import StandardFormat

    standard_format = StandardFormat()
    standard_format.set_image_name("000000000009.jpg")
    standard_format.set_image_dimensions(640, 480)
    standard_format.set_source_format('labelme')
    standard_format.set_source_bbox_format('pascal_voc')
    standard_format.add_bbox([0.31352486559139786, 0.6691868279569892, 0.4377352150537635, 0.9216173835125447], "test")

    manager = StandardSourceManager()
    labelme_data = manager.denormalize_to_source_data('labelme', standard_format)

    assert labelme_data.get('imagePath') == "000000000009.jpg"
    assert labelme_data.get('imageWidth') == 640
    assert labelme_data.get('imageHeight') == 480
    assert len(labelme_data.get('shapes')) == 1
    shape = labelme_data.get('shapes')[0]
    assert shape.get('label') == "test"
    assert shape.get('points') == [pytest.approx([60.58064516129033, 100.0215053763441]), pytest.approx([340.73118279569894, 542.39784946236557])]
    print(labelme_data)