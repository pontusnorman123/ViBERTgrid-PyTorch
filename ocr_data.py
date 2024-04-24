from paddleocr import PaddleOCR
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ocr = PaddleOCR(use_angle_cls=False,
                lang='sv',
                rec=False,
                det_db_score_mode='slow',
                rec_db_score_mode='slow',
                )  # need to run only once to download and load model into memory


def scale_bounding_box(box: list[int], width: float, height: float) -> list[int]:
    #print(box)
    return [
        1000 * box[0] / width,
        1000 * box[1] / height,
        (1000 * box[0] / width) + box[2],
        (1000 * box[1] / height) + box[3]
    ]



def process_bbox(box: list):
    return [box[0][0], box[1][1], box[2][0] - box[0][0], box[2][1] - box[1][1]]


def process_bbox(box: list):

    # Extract x and y coordinates separately
    x_coords = [point[0] for point in box]
    y_coords = [point[1] for point in box]

    # Find the min and max of x and y coordinates
    x_min = min(x_coords)
    y_min = min(y_coords)
    x_max = max(x_coords)
    y_max = max(y_coords)

    bounding_box = [x_min, y_min, x_max, y_max]
    #print("bbox: ", bounding_box)


    return bounding_box

def quad_to_box(quad):
    # test 87 is wrongly annotated
    box = (
        max(0, quad["x1"]),
        max(0, quad["y1"]),
        quad["x3"],
        quad["y3"]
    )
    if box[3] < box[1]:
        bbox = list(box)
        tmp = bbox[3]
        bbox[3] = bbox[1]
        bbox[1] = tmp
        box = tuple(bbox)
    if box[2] < box[0]:
        bbox = list(box)
        tmp = bbox[2]
        bbox[2] = bbox[0]
        bbox[0] = tmp
        box = tuple(bbox)
    return box

def normalize_bbox(bbox, size):
    return [
        int(100 * bbox[0] / size[0]),
        int(100 * bbox[1] / size[1]),
        int(100 * bbox[2] / size[0]),
        int(100 * bbox[3] / size[1]),
    ]


def dataSetFormat(img_file):
    width, height = img_file.size

    ress = ocr.ocr(np.asarray(img_file))

    test_dict = {'tokens': [], "bboxes": []}
    test_dict['img_path'] = img_file

    for item in ress[0]:
      print(item)
      normalized_bbox = normalize_bbox(process_bbox(item[0]), img_file.size)

      # Get the coordinates from the normalized bbox
      x0, y0, x1, y1 = normalized_bbox

      # Check if the bbox has a positive width and height
      if (y1 - y0 > 0) and (x1 - x0 > 0):
          test_dict['tokens'].append(item[1][0])
          test_dict['bboxes'].append(normalized_bbox)
      else:
          print("Invalid bbox with non-positive dimensions:", normalized_bbox)


    return test_dict["tokens"], test_dict["bboxes"]