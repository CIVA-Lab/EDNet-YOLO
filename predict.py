import argparse
import glob
import os

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from edf import EDF


def main():
    parser = argparse.ArgumentParser(description='EDF prediction.')
    parser.add_argument('-m', '--model_config_path', type=str,
                        default='config/yolov3.cfg')
    parser.add_argument('-c', '--model_ckpt_dir', type=str,
                        default='weights/edf')
    parser.add_argument('-d', '--data_root_dir', type=str, required=True)
    parser.add_argument('-s', '--sequences', nargs='+', default=[])
    parser.add_argument('-o', '--output_dir', type=str, default='output')

    args = parser.parse_args()

    model = EDF(args.model_config_path, args.model_ckpt_dir)

    sequences = args.sequences

    for sequence in sequences:
        print(f'{sequence:-^80}')
        # Process sequences
        sequence_dir = os.path.join(args.data_root_dir, sequence, '*')
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, sequence + '.txt')
        process_sequence(sequence_dir, model, output_path)


def process_sequence(sequence_dir, model, output_path):

    files = sorted(glob.glob(os.path.join(sequence_dir, '*.jpg')))
    H, W, _ = cv2.imread(files[0]).shape

    # Global tracking dataframe
    all_df = pd.DataFrame()

    for fr, f in enumerate(tqdm(files, desc='Processing'), 1):
        img = cv2.imread(f)

        boxes = model(img)

        N = len(boxes)

        boxes[:, [0, 2]] = np.round(boxes[:, [0, 2]].clip(0, W))
        boxes[:, [1, 3]] = np.round(boxes[:, [1, 3]].clip(0, H))
        boxes[:, [2, 3]] = boxes[:, [2, 3]] - boxes[:, [0, 1]]

        ids = np.arange(1, boxes.shape[0] + 1).reshape(-1, 1)
        frs = np.ones(boxes.shape[0]).reshape(-1, 1) * fr
        data = np.concatenate(
            [frs, ids, boxes[:, :5], np.ones((N, 2)) * -1], axis=1)

        # Add entries to global tracking table
        all_df = pd.concat([all_df, pd.DataFrame(data)])

    export_mot_df(all_df, output_path)
    print(f'Exported to {output_path}')


def export_mot_df(df, out_path):
    # Prepare dataframe
    all_df = df.astype({
        0: 'int',
        1: 'int',
        2: 'int',
        3: 'int',
        4: 'int',
        5: 'int',
        6: 'float',
        7: 'int',
        8: 'int',
    })

    all_df.to_csv(out_path, index=False, header=False)


if __name__ == '__main__':
    main()
