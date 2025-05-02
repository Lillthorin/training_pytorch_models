from scripts.train import train
from scripts.train_mask import train as train_masked

"Supported models: ssd_mobilenetv3, fasterrcnn_resnet50, 'fasterrcnn_mobile_320, fasterrcnn_mobile 'retinanet' "
"Supported schedulers: steplr, cosine, onecycle, reduceonplateau(mAP)" "DEFAULT == reduceonplateau "
"Supported optimizer: sgd, adam, adamw" "DEFAULT == adamw "
"Load model with pretrained backbone or without, set PRETRAINED_BACKBONE to False" "DEFAULT == True"
"Auto Augmentation AUGMENT = True, if you want to train without augmentation set to false" "DEFAULT == True"

train(DATA_DIR="", MODEL_NAME='ssd_mobilenetv3', BATCH_SIZE=4, EPOCHS=50, NUM_CLASSES=2, IMGSZ=640)
# train_masked(DATA_DIR="new_dataset", NUM_CLASSES=5, BATCH_SIZE=2,
# EPOCHS=10, LR=0.001, PRETRAINED_BACKBONE=True, SCHEDULER_NAME='cosine', OPTIMIZER_NAME='adamw', CONTINUE_TRAINING=True, CHECKPOINT_PATH=r'C:\Users\Thori\OneDrive\Desktop\Pytorch segmentering och ssd\testa_maskrcnn\runs\maskrcnn_resnet50\1\checkpoint_best_mask.pth')
