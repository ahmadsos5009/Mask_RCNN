import mrcnn.model as modellib

from TablesDataset import TablesDataset
from TablesConfig import TablesConfig

import os

MODEL_DIR = os.path.abspath('./')
COCO_MODEL_PATH = os.path.join(MODEL_DIR, 'mask_rcnn_coco.h5')

config = TablesConfig()

# Training dataset
dataset_train = TablesDataset()
dataset_train.load_dataset('TabStructDB', is_train=True)
dataset_train.prepare()
print('Train: %d' % len(dataset_train.image_ids))

# Validation dataset
dataset_val = TablesDataset()
dataset_val.load_dataset('TabStructDB', is_train=False)
dataset_val.prepare()
print('Validation: %d' % len(dataset_val.image_ids))

# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

model.load_weights(COCO_MODEL_PATH, by_name=True,
                   exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=1,
            layers='heads')
