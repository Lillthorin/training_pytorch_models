
# PyTorch Training Utilities

This repository contains utilities and scripts for training object detection models using PyTorch.

## About

Major parts of this code were generated or co-written with the help of ChatGPT and adapted for my own projects.  
It works for me – if it works for you too, great! If not, you're on your own. 😉 
I simply wanted to share this setup as it was tough for me to find something similar when first starting with Pytorch.

This setup was created with Python 3.10. Use requirements.txt to install used packages. This requirements.txt does not apply when using GPU.
Torch and Torchvision needs to be installed seperatly with GPU support in order to access the GPU. 
For Collab use requirements_collab.txt


## License

This project is licensed under the MIT License. See [LICENSE.txt](./LICENSE.txt) for full details.

**Note:**  
This project depends on third-party libraries (such as PyTorch, torchvision, albumentations, pandas, etc.) that each have their own licenses.  
It is your responsibility to comply with any third-party licenses if you use or distribute this project. 
See [THIRD_PARTY_LICENSES.txt](./THIRD_PARTY_LICENSES.txt) for full details.



## How to train a modell

### To start a training session on any of the supported models for object detection:

from scripts.train import train
  
train(DATA_DIR="", MODEL_NAME='ssd_mobilenetv3', BATCH_SIZE=4, EPOCHS=50, NUM_CLASSES=2, IMGSZ=640)

### To start a training session on masked_rcnn modell use this instead:

from scripts.train_mask import train as train_masked
  
train_masked(DATA_DIR="", BATCH_SIZE=4, EPOCHS=50, NUM_CLASSES=2, IMGSZ=640)

# WARNING: 
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
    CHECKPOINT_PATH: str = '' <---- Path to checkpoint to resume training. 
    EXPORT_TORCHSCRIPT: bool = False <---- DO NOT CHANGE! Working on setting this up. 
    WARMUP_EPOCHS: int = 3  <----- Number of warmup epochs, the modell will go from 1e-6 LR up to the set LR during these epochs. 
    

### Dataset structure
This training example uses coco.json annotation files to train the models. 

The training script expects the dataset to be structured as shown below. 
DATA_DIR="" in this example should be set to DATA_DIR="path-to/dataset"

![dataset_structure](https://github.com/user-attachments/assets/62ce90c6-bcc8-4412-ac90-3f4de05d7cfe)

### Outputs

During training a main folder will be created "runs" under this a subfolder with the current model name will be creating. Inside this folder every run will be listed as 1, 2, 3... 

To give the user some useful information a few outputs are created during the training: 

Under train_labels you can see how the images and labels are sent to the model, every time augmentation is changed 2 new images are created to illustrate how augmentation is applied and sent to the model:

![train_labels](https://github.com/user-attachments/assets/e12866a7-01c2-4f72-b4bc-d45d5fe9e13b)

Under the folder val you can see the validation from the 'best' model and the 'last' model. This is split into two images, one with labels and one with prediction.
Use this to track how your modell is doing: 
![best_val_pred](https://github.com/user-attachments/assets/49ac909f-e07c-494e-9d00-ad9c315aefd6)
![best_val_labels](https://github.com/user-attachments/assets/1d5d19aa-9fee-4175-b109-7c659d15fce6)


After the training has been completed a metrics grid will be created. THIS CHANGES BETWEEN maskrcnn modell and the rest:
![all_metrics_grid](https://github.com/user-attachments/assets/17e8ddc1-e3ea-47b4-878f-c5ed2ab6e091)


A confusion matrix is created as well. In the example below only one class (NUM_CLASSES=2) was used so the confusion matrix isnt valid:

![confusion_matrix_best](https://github.com/user-attachments/assets/40ddd200-33d9-41db-a395-4f3d44352840)





