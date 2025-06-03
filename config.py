DATA_DIR_0 = {
    "test":f"datasets/TFIDV/GASF/test", 
    "train":f"datasets/TFIDV/GASF/train",
    "val":f"datasets/TFIDV/GASF/val",
}

DATA_DIR_1 = {
    "test":f"datasets/TFIDVGADF/gaf_dataset1/test", 
    "train":f"datasets/TFIDVGADF/gaf_dataset1/train",
    "val":f"datasets/TFIDVGADF/gaf_dataset1/val",
}
DATA_DIR_2 = {
    "test":f"datasets/TFIDV/RPLOT/test", 
    "train":f"datasets/TFIDV/RPLOT/train",
    "val":f"datasets/TFIDV/RPLOT/val",
}



DATA_DIRS_GASF = [DATA_DIR_0]
DATA_DIRS_GADF = [DATA_DIR_1]
DATA_DIRS_RPLOT = [DATA_DIR_2]
DATA_DIRS = [DATA_DIRS_GASF, DATA_DIRS_GADF, DATA_DIRS_RPLOT]

BATCH_SIZE = 128
P = 0.2
WEIGHT_DECAY = 0.05
NUM_SAMPLES = 61574
NUM_EPOCHS = 100
RESOLUTION = 64
NUM_DATASETS = 1
UNFROZEN_LAYERS = [0,2,4]
