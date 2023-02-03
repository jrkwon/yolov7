# yolov7-pose

Note: This forked version focuses on how to train this pose estimation model.

## Code prep

Clone this repo.
```shell
git clone https://github.com/jrkwon/yolov7.git
```
Change the current directory and set the current branch to `yolo-pose`
```shell
cd yolov7
git checkout pose
```

## Data prep

Here is the structure of expected directories.
```
yolov7
│   README.md
│   ...   
│
coco
│   images
│   annotations
|   labels
│   └─────train2017
│       │       └───
|       |       └───
|       |       '
|       |       .
│       └─val2017
|               └───
|               └───
|               .
|               .
|    train2017.txt
|    val2017.txt
coco_kpts
│   images
│   annotations
|   labels
│   └─────train2017
│       │       └───
|       |       └───
|       |       '
|       |       .
│       └─val2017
|               └───
|               └───
|               .
|               .
|    train2017.txt
|    val2017.txt

```

Here is a step-by-step instruction to create this directory structure.

### `coco` directory prep

```shell
cd ..
wget https://github.com/ultralytics/yolov5/releases/download/v1.0/coco2017labels.zip
unzip coco2017labels.zip
```
Then, you will see `coco` directory and its structure is like below.
```
coco
│   images      <-- no image data    
│   annotations
|   labels
│   └─────train2017
│       │       └───
|       |       └───
|       |       '
|       |       .
│       └─val2017
|               └───
|               └───
|               .
|               .
|    train2017.txt
|    val2017.txt
```
`images` data download

Download and unzip train images
```shell
wget http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip -d coco/images
```

Download and unzip test images
```shell
wget http://images.cocodataset.org/zips/test2017.zip
unzip test2017.zip -d coco/images
```

Download [person_keypoints_val2017.json](https://drive.google.com/file/d/1nxs-w44m_V4XND0r3sho67XLeCdBNmtZ/view?usp=share_link)
Move the json file to `coco/annotations/`.

### `coco_kpts` directory prep

```shell
mkdir coco_kpts
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/coco2017labels-keypoints.zip
unzip coco2017labels-keypoints.zip -d coco_kpts
```

Then, you will see `coco_kpts` directory like below.
```shell
coco_kpts
|   labels
│   └─────train2017
│       │       └───
|       |       └───
|       |       '
|       |       .
│       └─val2017
|               └───
|               └───
|               .
|               .
|    train2017.txt
|    val2017.txt

```
`images` and `annotations` directories are necessary.
Instead of duplicate the folders, we can create symbolic links of them inside `coco_kpts`.

```shell
cd coco_kpts
ln -s ../coco/images/ images
ln -s ../coco/annotations/ annotations
cd ..
```

After this, you will see.
```shell
coco_kpts
│   images        <-- symbolic link to ../coco/images
│   annotations   <-- symbolic link to ../coco/annotations
|   labels
│   └─────train2017
│       │       └───
|       |       └───
|       |       '
|       |       .
│       └─val2017
|               └───
|               └───
|               .
|               .
|    train2017.txt
|    val2017.txt

```

### Pretrained modes for a person detector

```shell
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-person.pt -O weights/yolov7-w6-person.pt
```

## Train
```shell
python train.py --data data/coco_kpts.yaml --cfg cfg/yolov7-w6-pose.yaml --weights weights/yolov7-w6-person.pt --batch-size 64 --img 960 --kpt-label --name yolov7-w6-pose --sync-bn --hyp data/hyp.pose.yaml

```

You may need to reduce the batch size based on your GPU memory availability.

---

The below section is from the original.

---

# yolov7-pose
Implementation of "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors"

Pose estimation implimentation is based on [YOLO-Pose](https://arxiv.org/abs/2204.06806). 

## Dataset preparison

[[Keypoints Labels of MS COCO 2017]](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/coco2017labels-keypoints.zip)

## Training

[yolov7-w6-person.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-person.pt)

``` shell
python -m torch.distributed.launch --nproc_per_node 8 --master_port 9527 train.py --data data/coco_kpts.yaml --cfg cfg/yolov7-w6-pose.yaml --weights weights/yolov7-w6-person.pt --batch-size 128 --img 960 --kpt-label --sync-bn --device 0,1,2,3,4,5,6,7 --name yolov7-w6-pose --hyp data/hyp.pose.yaml
```

## Deploy
TensorRT:[https://github.com/nanmi/yolov7-pose](https://github.com/nanmi/yolov7-pose)

## Testing

[yolov7-w6-pose.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt)

``` shell
python test.py --data data/coco_kpts.yaml --img 960 --conf 0.001 --iou 0.65 --weights yolov7-w6-pose.pt --kpt-label
```

## Citation

```
@article{wang2022yolov7,
  title={{YOLOv7}: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors},
  author={Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark},
  journal={arXiv preprint arXiv:2207.02696},
  year={2022}
}
```

## Acknowledgements

<details><summary> <b>Expand</b> </summary>

* [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
* [https://github.com/WongKinYiu/yolor](https://github.com/WongKinYiu/yolor)
* [https://github.com/WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)
* [https://github.com/WongKinYiu/ScaledYOLOv4](https://github.com/WongKinYiu/ScaledYOLOv4)
* [https://github.com/Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
* [https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [https://github.com/DingXiaoH/RepVGG](https://github.com/DingXiaoH/RepVGG)
* [https://github.com/JUGGHM/OREPA_CVPR2022](https://github.com/JUGGHM/OREPA_CVPR2022)
* [https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose](https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose)

</details>
