DATA_DIR_0 = {
    "test":f"datasets/TFIDV/GASF0/test", 
    "train":f"datasets/TFIDV/GASF0/train",
    "val":f"datasets/TFIDV/GASF0/val",
}

DATA_DIR_1 = {
    "test":f"datasets/TFIDV/GASF1/test", 
    "train":f"datasets/TFIDV/GASF1/train",
    "val":f"datasets/TFIDV/GASF1/val",
}
DATA_DIR_2 = {
    "test":f"datasets/TFIDV/GASF2/test", 
    "train":f"datasets/TFIDV/GASF2/train",
    "val":f"datasets/TFIDV/GASF2/val",
}

DATA_DIR_3 = {
    "test":f"datasets/TFIDV/GADF0/test", 
    "train":f"datasets/TFIDV/GADF0/train",
    "val":f"datasets/TFIDV/GADF0/val",
}
DATA_DIR_4 = {
    "test":f"datasets/TFIDV/GADF1/test", 
    "train":f"datasets/TFIDV/GADF1/train",
    "val":f"datasets/TFIDV/GADF1/val",
}
DATA_DIR_5 = {
    "test":f"datasets/TFIDV/GADF2/test", 
    "train":f"datasets/TFIDV/GADF2/train",
    "val":f"datasets/TFIDV/GADF2/val",
}

DATA_DIR_6={
    "test":f"datasets/TFIDV/RPLOT0/test", 
    "train":f"datasets/TFIDV/RPLOT0/train",
    "val":f"datasets/TFIDV/RPLOT0/val",
}
DATA_DIR_7={
    "test":f"datasets/TFIDV/RPLOT1/test", 
    "train":f"datasets/TFIDV/RPLOT1/train",
    "val":f"datasets/TFIDV/RPLOT1/val",
}
DATA_DIR_8={
    "test":f"datasets/TFIDV/RPLOT2/test", 
    "train":f"datasets/TFIDV/RPLOT2/train",
    "val":f"datasets/TFIDV/RPLOT2/val",
}


DATA_DIRS_GASF = [DATA_DIR_0,DATA_DIR_1,DATA_DIR_2]
DATA_DIRS_GADF = [DATA_DIR_3,DATA_DIR_4,DATA_DIR_5]
DATA_DIRS_RPLOT = [DATA_DIR_6,DATA_DIR_7,DATA_DIR_8]

DATA_DIRS = [DATA_DIRS_GASF, DATA_DIRS_GADF, DATA_DIRS_RPLOT]

BATCH_SIZE = 128
P = 0.2
WEIGHT_DECAY = 0.05
NUM_SAMPLES = 61574
NUM_EPOCHS = 100
RESOLUTION = 64
NUM_DATASETS = 3
UNFROZEN_LAYERS = [0,4]
