NYU40_STUFF_CLASSES = [1, 2, 22]

NYU40_IGNORE_LABEL = 0

NYU40_THING_CLASSES = [i for i in range(41) if i not in NYU40_STUFF_CLASSES and i != NYU40_IGNORE_LABEL]

NYU40_NUM_CLASSES = 41

SCANNET_NYU40_EVALUATION_CLASSES = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    14,
    16,
    24,
    28,
    33,
    34,
    36,
    39,
]

NYU40_CLASS_IDS_TO_NAMES = {
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

NYU40_CLASS_IDS_TO_SIZES = {
    1: "M",
    2: "M",
    3: "M",
    4: "M",
    5: "M",
    6: "M",
    7: "M",
    8: "M",
    9: "M",
    10: "M",
    11: "M",
    12: "M",
    13: "M",
    14: "M",
    15: "M",
    16: "M",
    17: "M",
    18: "S",
    19: "M",
    20: "N",
    21: "S",
    22: "M",
    23: "M",
    24: "M",
    25: "S",
    26: "S",
    27: "S",
    28: "M",
    29: "M",
    30: "M",
    31: "M",
    32: "M",
    33: "M",
    34: "S",
    35: "S",
    36: "M",
    37: "S",
    38: "M",
    39: "M",
    40: "M",
}

PANOPTIC_LABEL_DIVISOR = 1000

NYU40_COLOR_PALETTE = [
    (0, 0, 0),
    (174, 199, 232),  # wall
    (152, 223, 138),  # floor
    (31, 119, 180),  # cabinet
    (255, 187, 120),  # bed
    (188, 189, 34),  # chair
    (140, 86, 75),  # sofa
    (255, 152, 150),  # table
    (214, 39, 40),  # door
    (197, 176, 213),  # window
    (148, 103, 189),  # bookshelf
    (196, 156, 148),  # picture
    (23, 190, 207),  # counter
    (178, 76, 76),
    (247, 182, 210),  # desk
    (66, 188, 102),
    (219, 219, 141),  # curtain
    (140, 57, 197),
    (202, 185, 52),
    (51, 176, 203),
    (200, 54, 131),
    (92, 193, 61),
    (78, 71, 183),
    (172, 114, 82),
    (255, 127, 14),  # refrigerator
    (91, 163, 138),
    (153, 98, 156),
    (140, 153, 101),
    (158, 218, 229),  # shower curtain
    (100, 125, 154),
    (178, 127, 135),
    (120, 185, 128),
    (146, 111, 194),
    (44, 160, 44),  # toilet
    (112, 128, 144),  # sink
    (96, 207, 209),
    (227, 119, 194),  # bathtub
    (213, 92, 176),
    (94, 106, 211),
    (82, 84, 163),  # otherfurn
    (100, 85, 144),
]
