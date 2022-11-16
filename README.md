# Ensemble PyTorch-YOLOv3
An implementation of Ensemble-YOLOv3 from: I.E. Toubal et al., 2022 "Ensemble 
Deep Learning Methods for Cell Detection and Tracking"

## Installation

This repository is based on the PyTorch implementation of YOLOv3 from
[eriklindernoren](https://github.com/eriklindernoren)
```bash
$ pip install git+https://github.com/CIVA-Lab/Ensemble-Detection-YOLOv3.git
$ pip install -r requirements.txt
```
## Download pretrained weights

CTMC pretrained weights can be downloaded from this Google Drive link. 
Alternatively, you can use the `gdown` python cli package:
```bash
$ pip install gdown
$ gdown 1ZI31NXaKWTSpq_ToLh_osO0qpjOiwvCB -O weights/weights.zip
$ unzip weights/weights.zip -d weights
$ rm weights/weights.zip # (Optional)
```

## Download CTMCv1 [2] dataset
```bash
$ curl 'https://motchallenge.net/data/CTMCV1.zip' -L -o 'CTMCV1.zip'
$ unzip 'CTMCV1.zip' -d data
$ rm 'CTMCV1.zip'
```

## Inference
To run inference, use the `predict.py` script. You can use the `--help` flag to
get a list of all available options. Below is an example
```bash
$ python predict.py --data_root_dir data/CTMCV1/test \
  --sequences 3T3-run02 3T3-run04 3T3-run06 3T3-run08 \
  --model_config_path cfg/yolov3.cfg \
  --model_ckpt_dir weights/edf \
  --output_dir output 
```

## Training
### Preprocess the data for YOLOv3 training
This part will split the training data into train/val splits in the format in
which this implementation of YOLOv3 is trained using the command:
```bash
$ python data_prep.py -i data/CTMCV1/train -o data/CTMC-prepped 
```
The downloaded data should be in the format:
```
data/CTMCV1/train/<sequence>
  - img1
    - 000001.jpg
    - 000002.jpg
    - ...
  - gt
    - gt.txt
```
Where `gt.txt` contains the cell bounding boxes stored in MOT format: `frame, id, x, y, w, h, conf, -1,-1`. In this format, `x`, `y` are the location coordinates of the top left of the bounding box.

We format this data to the following structure:
```
data/CTMC-prepped
  - images
    - <sequence>-000001.jpg
    - <sequence>-000002.jpg
    - ...
  - labels
    - <sequence>-000001.txt
    - <sequence>-000002.txt
    - ...
  train.txt
  val.txt
  classes.names
  data.data
```
### Load ImageNet weights (Optional)
Download weights for backbone network:
```bash
$ wget -c "https://pjreddie.com/media/files/darknet53.conv.74" \
  --header "Referer: pjreddie.com" \
  --output-document weights/darknet53.conv.74
```
### Train the model

For argument descriptions have a look at `yolo-train --help`

### Example 
To train on CTMC using a Darknet-53 backend pretrained on ImageNet run:

```bash
$ yolo-train \
    --data data/CTMC-prepped/data.data \
    --pretrained_weights weights/darknet53.conv.74 \
    --checkpoint_interval 20 \
    --evaluation_interval 1 \
    --checkpoint_path checkpoints \
    --logdir logs/ctmc \
    --n_cpu 32
```

### Tensorboard
Track training progress in Tensorboard:
* Initialize training
* Run the command below
* Go to http://localhost:6006/

```bash
$ tensorboard --logdir='logs' --port=6006
```

Storing the logs on a slow drive possibly leads to a significant training speed
decrease.

You can adjust the log directory using `--logdir <path>` when running
`tensorboard` and `yolo-train`.

## References

- [1] Original repository for YOLOv3 Pytorch:
  https://github.com/eriklindernoren/PyTorch-YOLOv3

- [2] Anjum, S., & Gurari, D. (2020). CTMC: Cell Tracking With Mitosis Detection 
    Dataset Challenge. In Proceedings of the IEEE/CVF Conference on Computer 
    Vision and Pattern Recognition Workshops (pp. 982-983).

- [3] Redmon, J., & Farhadi, A. (2018). Yolov3: An incremental improvement. arXiv 
    preprint arXiv:1804.02767.

## Cite this paper
```
@article{toubal2022ensemble,
  title={Ensemble Deep Learning Methods for Cell Detection and Tracking},
  author={Toubal, Imad Eddine and Alshakarji, Noor and Palaniappan, K.},
  submitted={OJEMB},
  year={2022}
}
```