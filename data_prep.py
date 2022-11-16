import argparse
import glob
import os
import shutil

import numpy as np
import pandas as pd
import tqdm
from sklearn.model_selection import train_test_split


def bbox_to_yolo(bbox, w, h):
  """Converts bounding boxes to the YOLO format:
  [x1, y1, w, h, ...] => [cx, cy, w, h]

  Args:
      bbox (np.ndarray): Input bounding boxes in MOT format:
          [x1, y1, w, h,...].
      w (int): Image width.
      h (int): Image height.

  Returns:
      np.ndarray: Bounding boxes in YOLO format. [cx, cy, w, h]
  """
  cx = (bbox[:, [1]] + (bbox[:, [3]] / 2)) / w
  cy = (bbox[:, [2]] + (bbox[:, [4]] / 2)) / h
  ww = bbox[:, [3]] / w
  hh = bbox[:, [4]] / h

  return np.concatenate([bbox[:, [0]], cx, cy, ww, hh], axis=1)


def read_ini(path):
  """Reads an ini file and returns a dictionary with the contents.

  Args:
    path (str): Path to the ini file.

  Returns:
    dict[str, str]: Dictionary with the contents of the ini file.
  """
  with open(path) as f:
    lines = f.readlines()
  lines = [line.strip() for line in lines]
  lines = [line for line in lines if line and not line.startswith('#')]
  lines = [line.split('=') for line in lines]
  lines = [line for line in lines if len(line) == 2]
  return {line[0].strip(): line[1].strip() for line in lines}


def main(args):
  folders = glob.glob(os.path.join(args.images_dir, '*'))
  iterator = tqdm.tqdm(folders)

  images_dir = os.path.join(args.target_dir, 'images')
  labels_dir = os.path.join(args.target_dir, 'labels')

  os.makedirs(images_dir, exist_ok=True)
  os.makedirs(labels_dir, exist_ok=True)

  for folder in iterator:
    basename = os.path.basename(folder)
    iterator.set_description(f'Processing {basename}')

    # Sequence info:
    ini_file = os.path.join(folder, 'seqinfo.ini')
    info = read_ini(ini_file)
    h, w = int(info['imHeight']), int(info['imWidth'])
    ext = info['imExt']
    img_dir = info['imDir']
    seq_length = int(info['seqLength'])

    # Cell type and sequence number:
    cell_type, _, seq = basename.rpartition('-run')

    # 1. Create sylinks to images.
    images_pattern = os.path.join(folder, img_dir, f'*{ext}')
    for frame in glob.glob(images_pattern):
      frame_name = os.path.basename(frame)
      out_name = f'{cell_type}-{seq}-{frame_name}'
      try:
        shutil.copy(frame, os.path.join(images_dir, out_name))
      except FileExistsError:
        pass

    # 2. Convert bbox format to YOLO format.
    bboxes = np.loadtxt(os.path.join(
        folder, 'gt', 'gt.txt'), delimiter=',')
    frames = bboxes[:, 0].astype('int')
    boxes = bboxes[:, 1:6]
    boxes[:, 0] = 0
    for i in range(1, seq_length + 1):
      frame_boxes = boxes[frames == i]
      frame_boxes = bbox_to_yolo(frame_boxes, w, h)
      out_path = os.path.join(
          labels_dir, f'{cell_type}-{seq}-{i:06d}.txt')

      df = pd.DataFrame(frame_boxes, columns=['class', 'x', 'y', 'w', 'h'])
      df = df.astype({'class': 'int'})

      df.to_csv(out_path, sep=' ', index=False, header=False)

  # 3. Create train.txt and val.txt files.
  image_files = glob.glob(os.path.join(images_dir, f'*{ext}'))
  train_files, valid_files = sample(image_files, args.sampling_strategy)

  train_txt = os.path.join(args.target_dir, 'train.txt')
  val_txt = os.path.join(args.target_dir, 'val.txt')
  classes_names = os.path.join(args.target_dir, 'classes.names')

  with open(train_txt, 'w') as f:
    print(f'Writing {train_txt}')
    f.writelines([f + '\n' for f in train_files])

  with open(val_txt, 'w') as f:
    print(f'Writing {val_txt}')
    f.writelines([f + '\n' for f in valid_files])

  # This file is used for multi-class detection, we don't care for it but
  # it needs to be provided for training.
  with open(classes_names, 'w') as f:
    print(f'Writing {classes_names}')
    f.writelines(['cell'])

  # 4. Finally, put it all together in a .data file:
  with open(os.path.join(args.target_dir, 'data.data'), 'w') as f:
    print(f'Writing {args.target_dir}/data.data')
    f.write(f'classes={1}\n')
    f.write(f'train={train_txt}\n')
    f.write(f'valid={val_txt}\n')
    f.write(f'names={classes_names}\n')


def sample(all_files, strategy, valid_size=0.2, val_cell_type='3T3'):
  """Samples training and testing splits from a list of images using a
  certain sampling strategy.

  Args:
    all_files (list[str]): List of all image files.
    strategy (str): Sampling strategy. One of 'random_sampling',
    'leave_one_cell_type_out', 'leave_one_sequence_out'.
    valid_size (float, optional): Only used if 'strategy' is set to
    'random_sampling'. Represents the fraction of the full set that will
    be used for validation. Defaults to 0.2.
    val_cell_type (str, optional): Only used if 'strategy' is set to
    'leave_one_cell_type_out'. Represents the cell type that will be used
    for validation. Defaults to '3T3'.

  Raises:
    ValueError: if 'strategy' is not one of the supported values.

  Returns:
    tuple(list[str], list[str]): List of training files and list of
    validation files.
  """
  print(strategy)
  assert strategy in ['random_sampling', 'leave_one_cell_type_out',
                      'leave_one_sequence_out']

  if strategy == 'random_sampling':
    train_files, valid_files = train_test_split(
        all_files, test_size=valid_size)
  elif strategy == 'leave_one_cell_type_out':
    train_files = [f for f in all_files if val_cell_type not in f]
    valid_files = [f for f in all_files if val_cell_type in f]
  elif strategy == 'leave_one_sequence_out':
    # Sequence 01 will be removed from training
    train_files = [
        f for f in all_files if '-01-' not in os.path.basename(f)]
    valid_files = [f for f in all_files if '-01-' in os.path.basename(f)]
  else:
    raise ValueError('Unknown sampling strategy')
  return train_files, valid_files


if __name__ == '__main__':
  parser = argparse.ArgumentParser('Prepare data for training and inference '
                                   'of YOLOv3.')
  parser.add_argument('-i', '--images_dir', type=str, required=True,
                      help='Directory of the CTMC training data.')
  parser.add_argument('-o', '--target_dir', type=str, required=True,
                      help='Directory to store the prepped data.')
  parser.add_argument('-s', '--sampling_strategy', type=str,
                      choices=['random_sampling', 'leave_one_cell_type_out',
                               'leave_one_sequence_out'],
                      default='leave_one_cell_type_out',
                      help='Sampling strategy for train/val split.')

  main(parser.parse_args())
