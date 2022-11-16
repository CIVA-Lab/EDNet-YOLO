import glob
import os

import numpy as np
import torch

from pytorchyolo import detect, models


def coalesce_boxes(ensemble_bboxes):
  ensemble_bboxes = sorted(ensemble_bboxes, key=lambda x: x[4])

  merged_boxes = []

  while(len(ensemble_bboxes) > 0):
    current_list = [ensemble_bboxes.pop()]
    for i, other in reversed(list(enumerate(ensemble_bboxes))):
      if bb_intersection_over_union(current_list[0], other) > 0.5:
        current_list.append(other)
        # remove other
        ensemble_bboxes = ensemble_bboxes[:i] + ensemble_bboxes[i + 1:]

    merged_boxes.append(current_list)

  return merged_boxes


def xyxy2xywh(x):
  # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where
  # xy1=top-left, xy2=bottom-right
  y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
  y[0] = (x[0] + x[2]) / 2  # x center
  y[1] = (x[1] + x[3]) / 2  # y center
  y[2] = x[2] - x[0]  # width
  y[3] = x[3] - x[1]  # height
  return y


def xywh2xyxy(x):
  # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where
  # xy1=top-left, xy2=bottom-right
  y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
  y[0] = x[0] - x[2] / 2  # top left x
  y[1] = x[1] - x[3] / 2  # top left y
  y[2] = x[0] + x[2] / 2  # bottom right x
  y[3] = x[1] + x[3] / 2  # bottom right y
  return y


def bb_intersection_over_union(boxA, boxB):
  # xywh2xyxy

  # determine the (x, y)-coordinates of the intersection rectangle
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])

  # compute the area of intersection rectangle
  interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

  # compute the area of both the prediction and ground-truth
  # rectangles
  boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
  boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

  # compute the intersection over union by taking the intersection
  # area and dividing it by the sum of prediction + ground-truth
  # areas - the interesection area
  iou = interArea / float(boxAArea + boxBArea - interArea)

  # return the intersection over union value
  return iou


def nms(boxes, overlapThresh):
  # if there are no boxes, return an empty list
  if len(boxes) == 0:
    return [0]
  # initialize the list of picked indexes
  pick = []
  probFinal = 0
  # grab the coordinates of the bounding boxes
  x1 = boxes[:, 1].astype(float)
  y1 = boxes[:, 2].astype(float)
  x2 = boxes[:, 3].astype(float)
  y2 = boxes[:, 4].astype(float)
  prob = boxes[:, 5].astype(float)
  for val in prob:
    probFinal = probFinal + val
  # compute the area of the bounding boxes and sort the bounding
  # boxes by the bottom-right y-coordinate of the bounding box
  area = (x2 - x1 + 1) * (y2 - y1 + 1)
  # idxs = np.argsort(-prob)
  idxs = np.argsort(y2)
  # keep looping while some indexes still remain in the indexes
  # list
  while len(idxs) > 0:
    # grab the last index in the indexes list, add the index
    # value to the list of picked indexes, then initialize
    # the suppression list (i.e. indexes that will be deleted)
    # using the last index
    last = len(idxs) - 1
    i = idxs[last]
    pick.append(i)
    suppress = [last]
    # loop over all indexes in the indexes list
    for pos in range(0, last):
      # grab the current index
      j = idxs[pos]

      # find the largest (x, y) coordinates for the start of
      # the bounding box and the smallest (x, y) coordinates
      # for the end of the bounding box
      xx1 = max(x1[i], x1[j])
      yy1 = max(y1[i], y1[j])
      xx2 = min(x2[i], x2[j])
      yy2 = min(y2[i], y2[j])

      # compute the width and height of the bounding box
      w = max(0, xx2 - xx1 + 1)
      h = max(0, yy2 - yy1 + 1)

      # compute the ratio of overlap between the computed
      # bounding box and the bounding box in the area list
      overlap = float(w * h) / area[j]

      # if there is sufficient overlap, suppress the
      # current bounding box
      if overlap > overlapThresh:
        suppress.append(pos)

    # delete all indexes from the index list that are in the
    # suppression list
    idxs = np.delete(idxs, suppress)
  # return only the bounding boxes that were picked
  return boxes[pick], probFinal


class EDF(torch.nn.Module):
  def __init__(self, model_config_path, models_dir,
               ensemble_option='consensus'):
    """Load all checkpoints into a single ensemble model.
    Args:
      model_config_path (str): Path to model config file.
      models_dir (str): Path to directory containing model checkpoints.
      ensemble_option (str): Ensemble option. One of 'consensus',
        'affirmative', or 'unanymous'.
    """
    super(EDF, self).__init__()
    checkpoints = glob.glob(os.path.join(models_dir, '*.pth'))
    self.models = []
    for checkpoint in checkpoints:
      print('Loading checkpoint', checkpoint)
      model = models.load_model(model_config_path, checkpoint)
      self.models.append(model)
    self.ensemble_option = ensemble_option

  @property
  def num_models(self):
    return len(self.models)

  def __call__(self, im: np.ndarray):
    H, W = im.shape[:2]
    all_boxes = []
    for model in self.models:
      group = detect.detect_image(model, im, conf_thres=.3, nms_thres=.7)

      group[:, [0, 2]] = np.round(group[:, [0, 2]].clip(0, W))
      group[:, [1, 3]] = np.round(group[:, [1, 3]].clip(0, H))
      all_boxes.extend(group)

    # Single pool of boxes array([[x1, y1, x2, y2, conf, cls], ...]) to
    # grouped by proximity [array([[x1, y1, x2, y2, conf, cls], ...]), ...]
    box_groups = coalesce_boxes(all_boxes)
    pick = []
    result = []

    for group in box_groups:
      list1 = []

      for rc in group:
        list1.append(rc)
      pick = []

      if self.ensemble_option == 'consensus':
        if len(np.array(list1)) >= self.num_models / 3:
          # list1 = np.array(list1)
          # pick = list1[[np.argmax(list1[:,5])]]
          pick, prob = nms(np.array(list1), 0.7)
          pick[0][5] = prob / self.num_models

      elif self.ensemble_option == 'unanimous':
        if len(np.array(list1)) == self.num_models:
          pick, prob = nms(np.array(list1), 0.7)
          pick[0][5] = prob / self.num_models

      elif self.ensemble_option == 'affirmative':
        pick, prob = nms(np.array(list1), 0.7)
        pick[0][5] = prob / self.num_models

      if len(pick) != 0:
        result.append(list(pick[0]))

    boxes = np.array(result)
    return boxes
