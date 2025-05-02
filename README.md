
# PyTorch Training Utilities

This repository contains utilities and scripts for training object detection models using PyTorch.

## About

Major parts of this code were generated or co-written with the help of ChatGPT and adapted for my own projects.  
It works for me – if it works for you too, great! If not, you're on your own. 😉 
I simply wanted to share this setup as it was tough for me to find something similar when first starting with Pytorch.

## License

This project is licensed under the MIT License. See [LICENSE.txt](./LICENSE.txt) for full details.

**Note:**  
This project depends on third-party libraries (such as PyTorch, torchvision, albumentations, pandas, etc.) that each have their own licenses.  
It is your responsibility to comply with any third-party licenses if you use or distribute this project. 



## How to train a modell

To start a training session on any of the supported models for object detection:

from scripts.train import train
  
train(DATA_DIR="", MODEL_NAME='ssd_mobilenetv3', BATCH_SIZE=4, EPOCHS=50, NUM_CLASSES=2, IMGSZ=640)

To start a training session on masked_rcnn modell use this instead:

from scripts.train_mask import train as train_masked
  
train_masked(DATA_DIR="", BATCH_SIZE=4, EPOCHS=50, NUM_CLASSES=2, IMGSZ=640)

WARNING: 
NUM_CLASSES is always 1 more than your number of classes. The models counts bakground as a class. In the json file you always have to have a supercategory:

categories":[{"id":0,"name":"bkgd","supercategory":"none"}, {"id":0,"name":"YOUR_FIRST_CLASS","supercategory":"bkgd".....}]

### Training settings


"Supported models: ssd_mobilenetv3, fasterrcnn_resnet50, 'fasterrcnn_mobile_320, fasterrcnn_mobile 'retinanet' "
"Supported schedulers: steplr, cosine, onecycle, reduceonplateau(mAP)" "DEFAULT == reduceonplateau "
"Supported optimizer: sgd, adam, adamw" "DEFAULT == adamw "
"Load model with pretrained backbone or without, set PRETRAINED_BACKBONE to False" "DEFAULT == True"
"Auto Augmentation AUGMENT = True, if you want to train without augmentation set to false" "DEFAULT == True"


    DATA_DIR: Any   <---- Path to your dataset, must be entered
    MODEL_NAME: Any   < --- choose model (valid inputs: ssd_mobilenetv3, fasterrcnn_resnet50, 'fasterrcnn_mobile_320, fasterrcnn_mobile 'retinanet')
    EPOCHS: Any    <---- How many epochs to train 
    NUM_CLASSES: Any  <------ Your number of classes +1 
    BATCH_SIZE = 4  <---- Default set to 4, increase if your gpu can handle it.
    IMGSZ: int = 640  <----- In this training setup the image size is fixed. Please set your desired image size. DEFAULT is 640
    LR: float = 0.001    <---- Learning rate experiment with this as you please. 
    AUGMENT: bool = True  <---- Aumentation is applied automatically, set to false to not apply augmentation. Augmentation is applied in 3 stages HEAVY, LOW, NO 
    OPTIMIZER_NAME: str = 'adamw' <---- Supported optimizer: sgd, adam, adamw: DEFAULT == adamw
    PRETRAINED_BACKBONE: bool = True <--- This applies a pretrained backbone to the model, if changed you are on your own. 
    SCHEDULER_NAME: str = 'reduceonplateau' <---Supported schedulers: steplr, cosine, onecycle, reduceonplateau(mAP)" "DEFAULT == reduceonplateau
    PATIENCE: int = 10  <--- Patience set for early stopping trigger (mAP)
    RESUME_TRAINING: bool = False <---- To resume training on a model set RESUME_TRAINING to True and apply checkpoint path to checkpoint_best.pth or checkpoint_last.pth
    CHECKPOINT_PATH: str = ''
    EXPORT_TORCHSCRIPT: bool = False
    WARMUP_EPOCHS: int = 3
    

### Dataset structure
This training example uses coco.json annotation files to train the models. 
The training script expects the dataset to be structured as shown below. 
DATA_DIR="" in this example should be set to DATA_DIR="path-to/dataset"

![dataset_structure](https://github.com/user-attachments/assets/62ce90c6-bcc8-4412-ac90-3f4de05d7cfe)

