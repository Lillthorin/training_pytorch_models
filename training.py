from scripts.train import train
from scripts.train_mask import train as train_masked

"Supported models: ssd_mobilenetv3, fasterrcnn_resnet50, 'fasterrcnn_mobile_320, fasterrcnn_mobile 'retinanet' "
"Supported schedulers: steplr, cosine, onecycle, reduceonplateau(mAP)" "DEFAULT == reduceonplateau "
"Supported optimizer: sgd, adam, adamw" "DEFAULT == adamw "
"Load model with pretrained backbone or without, set PRETRAINED_BACKBONE to False" "DEFAULT == True"
"Auto Augmentation AUGMENT = True, if you want to train without augmentation set to false" "DEFAULT == True"

train(DATA_DIR="", MODEL_NAME='ssd_mobilenetv3', BATCH_SIZE=4, EPOCHS=50, NUM_CLASSES=2, IMGSZ=320)
train_masked(DATA_DIR="t", NUM_CLASSES=2, BATCH_SIZE=4, EPOCHS=50, IMGSZ=640)
