NYU40_STUFF_CLASSES = [1, 2, 22]

NYU40_THING_CLASSES = [i for i in range(41) if i not in NYU40_STUFF_CLASSES]

NYU40_IGNORE_LABEL = 0

MAP_NYU40_CLASS_IDS_TO_NAMES = {
    1: "wall",
    2: "floor",
    3: "cabinet",
    4: "bed",
    5: "chair",
    6: "sofa",
    7: "table",
    8: "door",
    9: "window",
    10: "bookshelf",
    11: "picture",
    12: "counter",
    13: "blinds",
    14: "desk",
    15: "shelves",
    16: "curtain",
    17: "dresser",
    18: "pillow",
    19: "mirror",
    20: "floor mat",
    21: "clothes",
    22: "ceiling",
    23: "books",
    24: "refridgerator",
    25: "television",
    26: "paper",
    27: "towel",
    28: "shower curtain",
    29: "box",
    30: "whiteboard",
    31: "person",
    32: "nightstand",
    33: "toilet",
    34: "sink",
    35: "lamp",
    36: "bathtub",
    37: "bag",
    38: "otherstructure",
    39: "otherfurniture",
    40: "otherprop",
}

PANOPTIC_LABEL_DIVISOR = 1000