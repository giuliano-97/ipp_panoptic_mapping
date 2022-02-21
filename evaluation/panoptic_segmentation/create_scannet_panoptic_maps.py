import argparse
import functools
import os
import logging
import shutil
import zipfile
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

from utils.common import PANOPTIC_LABEL_DIVISOR, NYU40_STUFF_CLASSES

_SEMANTIC_MAPS_ARCHIVE_SUFFIX = "_2d-label-filt.zip"
_INSTANCE_MAPS_ARCHIVE_SUFFIX = "_2d-instance-filt.zip"
_SEMANTIC_MAPS_DIR_NAME = "label-filt"
_INSTANCE_MAPS_DIR_NAME = "instance-filt"
_PANOPTIC_MAPS_DIR_NAME = "panoptic"


def _scan_has_panoptic(scan_dir_path: Path):
    panoptic_maps_dir_path = scan_dir_path / _PANOPTIC_MAPS_DIR_NAME
    if panoptic_maps_dir_path.exists() and any(panoptic_maps_dir_path.iterdir()):
        return True

    panoptic_maps_archive_path = panoptic_maps_dir_path.with_suffix(".tar.gz")
    if panoptic_maps_archive_path.exists():
        return True

    return False


def _extract_zip_archive(path_to_zip_archive: Path):
    archive = zipfile.ZipFile(str(path_to_zip_archive))
    extract_dir = path_to_zip_archive.parent
    archive.extractall(str(extract_dir))


def _convert_semantic_map_labels(
    semantic_map: np.ndarray,
    label_conversion_dict: Dict,
):
    return np.vectorize(label_conversion_dict.get)(semantic_map)


def _normalize_instance_map(
    instance_map: np.ndarray, semantic_map: np.array, stuff_classes: List[int]
):
    normalized_instance_map = np.zeros_like(instance_map)

    for class_id in np.unique(semantic_map):
        if class_id == 0 or class_id in stuff_classes:
            continue
        # Get mask for the current class
        class_mask = semantic_map == class_id

        # Get all the unique instance ids
        instance_ids = np.unique(instance_map[class_mask]).tolist()

        # Remove 0 just in case
        try:
            instance_ids.remove(0)
        except ValueError:
            pass

        # Remap instance ids so they are 1-indexed
        new_instance_ids = list(range(1, len(instance_ids) + 1))

        # Create new instance
        for i, old_id in enumerate(instance_ids):
            instance_mask = np.logical_and(class_mask, instance_map == old_id)
            normalized_instance_map[instance_mask] = new_instance_ids[i]

    return normalized_instance_map


def _make_panoptic_from_semantic_and_instance(
    semantic_map: np.ndarray,
    instance_map: np.ndarray,
):
    panoptic_map = semantic_map * PANOPTIC_LABEL_DIVISOR + instance_map
    return panoptic_map.astype(np.int32)


def generate_deeplab2_panoptic_map(
    semantic_map_file_path: Path,
    instance_map_file_path: Path,
    panoptic_maps_dir_path: Path,
):
    semantic_map = np.array(Image.open(str(semantic_map_file_path)))
    instance_map = np.array(Image.open(str(instance_map_file_path)))

    # Convert semantic labels to the target labels set
    semantic_map = _convert_semantic_map_labels(
        semantic_map,
        SCANNETV2_TO_NYU40,
    )

    instance_map = _normalize_instance_map(
        instance_map,
        semantic_map,
        NYU40_STUFF_CLASSES,
    )

    # Make panoptic map
    panoptic_map = _make_panoptic_from_semantic_and_instance(
        semantic_map,
        instance_map,
    )

    # Save panoptic map to disk
    panoptic_map_file_path = panoptic_maps_dir_path / (
        semantic_map_file_path.stem.zfill(5) + ".png"
    )
    panoptic_map_image = Image.fromarray(panoptic_map)
    panoptic_map_image.save(str(panoptic_map_file_path))


def _create_panoptic_maps_for_scan(
    scan_dir_path: Path,
    remove_semantic_and_instance: bool,
    compress: bool,
):
    # Check if panoptic maps have already been created for this scans
    if _scan_has_panoptic(scan_dir_path):
        logging.warning(f"{scan_dir_path.name} already has panoptic!")
        return
    panoptic_maps_dir_path = scan_dir_path / _PANOPTIC_MAPS_DIR_NAME
    panoptic_maps_dir_path.mkdir(exist_ok=True)

    semantic_maps_dir_path = scan_dir_path / _SEMANTIC_MAPS_DIR_NAME
    instance_maps_dir_path = scan_dir_path / _INSTANCE_MAPS_DIR_NAME
    remove_semantic = False or remove_semantic_and_instance
    remove_instance = False or remove_semantic_and_instance
    if not semantic_maps_dir_path.exists():
        # If not found, try to extract the archive
        semantic_maps_archive_path = scan_dir_path / (
            scan_dir_path.stem + _SEMANTIC_MAPS_ARCHIVE_SUFFIX
        )
        if not semantic_maps_archive_path.exists():
            logging.warning(
                '"label-filt" missing in scan {}. Skipped.'.format(
                    str(scan_dir_path.name)
                )
            )
            return
        _extract_zip_archive(semantic_maps_archive_path)
        remove_semantic = True
    if not instance_maps_dir_path.exists():
        instance_maps_archive_path = scan_dir_path / (
            scan_dir_path.stem + _INSTANCE_MAPS_ARCHIVE_SUFFIX
        )
        if not instance_maps_archive_path.exists():
            logging.warning(
                '"instance-filt" missing in scan {}. Skipped.'.format(
                    str(scan_dir_path.name)
                )
            )
            return
        _extract_zip_archive(instance_maps_archive_path)
        remove_instance = True

    semantic_map_files = sorted(list(semantic_maps_dir_path.glob("*.png")))
    instance_map_files = sorted(list(instance_maps_dir_path.glob("*.png")))

    # Generate panoptic maps
    for semantic_map_file_path, instance_map_file_path in zip(
        semantic_map_files, instance_map_files
    ):
        generate_deeplab2_panoptic_map(
            semantic_map_file_path,
            instance_map_file_path,
            panoptic_maps_dir_path,
        )

    # Delete semantic and instance maps
    if remove_semantic:
        shutil.rmtree(semantic_maps_dir_path)

    if remove_instance:
        shutil.rmtree(instance_maps_dir_path)

    # Compress panoptic maps into a tar.gz archive
    if compress:
        cmd = f"cd {str(scan_dir_path.absolute())} && tar --remove-files -czf {_PANOPTIC_MAPS_DIR_NAME}.tar.gz {_PANOPTIC_MAPS_DIR_NAME}"
        os.system(cmd)


def _main(
    scans_root_dir_path: Path,
    remove_semantic_and_instance: bool,
    compress: bool,
    n_jobs: int,
):
    # Validate input args
    assert scans_root_dir_path.exists()
    assert n_jobs > 0

    # Get all the scan dirs
    scan_dir_paths = [
        p for p in sorted(list(scans_root_dir_path.glob("scene*"))) if p.is_dir()
    ]

    if n_jobs > 1:
        # Create panoptic maps for every directory in parallel
        job_fn = functools.partial(
            _create_panoptic_maps_for_scan,
            remove_semantic_and_instance=remove_semantic_and_instance,
            compress=compress,
        )
        with ThreadPool(n_jobs) as p:
            p.map(job_fn, scan_dir_paths)
    else:
        for scan_dir_path in scan_dir_paths:
            _create_panoptic_maps_for_scan(
                scan_dir_path=scan_dir_path,
                remove_semantic_and_instance=remove_semantic_and_instance,
                compress=compress,
            )


def _parse_args():

    parser = argparse.ArgumentParser(
        description="Create groundtruth panoptic maps for ScanNet scans."
    )
    parser.add_argument(
        "scans_root_dir_path",
        type=lambda p: Path(p).absolute(),
        help="Path to a directory containing scans as subdirs.",
    )

    parser.add_argument(
        "--remove_semantic_and_instance",
        action="store_true",
        help="Semantic and instance labels will be removed.",
    )

    parser.add_argument(
        "--compress",
        action="store_true",
        help="Panoptic maps will be compressed into a tar.gz archive.",
    )

    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=2,
        help="Number of parallel jobs",
    )

    return parser.parse_args()


SCANNETV2_TO_NYU40 = {
    0: 0,
    1: 1,
    2: 5,
    22: 23,
    3: 2,
    5: 8,
    1163: 40,
    16: 9,
    4: 7,
    56: 39,
    13: 18,
    15: 11,
    41: 22,
    26: 29,
    161: 8,
    19: 40,
    7: 3,
    9: 14,
    8: 15,
    10: 5,
    31: 27,
    6: 6,
    14: 34,
    48: 40,
    28: 35,
    11: 4,
    18: 10,
    71: 19,
    21: 16,
    40: 40,
    52: 30,
    96: 39,
    29: 3,
    49: 40,
    23: 5,
    63: 40,
    24: 7,
    17: 33,
    47: 37,
    32: 21,
    46: 40,
    65: 40,
    97: 39,
    34: 32,
    38: 40,
    33: 25,
    75: 3,
    36: 17,
    64: 40,
    101: 40,
    130: 40,
    27: 24,
    44: 7,
    131: 40,
    55: 28,
    42: 36,
    59: 40,
    159: 12,
    74: 5,
    82: 40,
    1164: 3,
    93: 40,
    77: 40,
    67: 39,
    128: 1,
    50: 40,
    35: 12,
    69: 38,
    100: 40,
    62: 38,
    105: 38,
    1165: 1,
    165: 24,
    76: 40,
    230: 40,
    54: 40,
    125: 38,
    72: 40,
    68: 39,
    145: 38,
    157: 40,
    1166: 40,
    132: 40,
    1167: 8,
    232: 38,
    134: 40,
    51: 39,
    250: 40,
    1168: 38,
    342: 38,
    89: 38,
    103: 40,
    99: 39,
    95: 38,
    154: 38,
    140: 20,
    1169: 39,
    193: 38,
    116: 39,
    202: 40,
    73: 40,
    78: 38,
    1170: 40,
    79: 26,
    80: 31,
    141: 38,
    57: 39,
    102: 40,
    261: 40,
    118: 40,
    136: 38,
    98: 40,
    1171: 38,
    170: 40,
    1172: 40,
    1173: 3,
    221: 40,
    570: 37,
    138: 40,
    168: 40,
    276: 8,
    106: 40,
    214: 40,
    323: 40,
    58: 38,
    86: 13,
    399: 40,
    121: 40,
    185: 40,
    300: 40,
    180: 40,
    163: 40,
    66: 40,
    208: 40,
    112: 40,
    540: 29,
    395: 38,
    166: 40,
    122: 39,
    120: 38,
    107: 38,
    283: 40,
    88: 40,
    90: 39,
    177: 39,
    1174: 40,
    562: 40,
    1175: 40,
    1156: 12,
    84: 38,
    104: 39,
    229: 40,
    70: 39,
    325: 40,
    169: 40,
    331: 40,
    87: 39,
    488: 40,
    776: 40,
    370: 40,
    191: 38,
    748: 40,
    242: 40,
    45: 7,
    417: 2,
    188: 38,
    1176: 40,
    1177: 39,
    1178: 38,
    110: 39,
    148: 40,
    155: 39,
    572: 40,
    1179: 38,
    392: 40,
    1180: 39,
    609: 38,
    1181: 40,
    195: 40,
    581: 39,
    1182: 40,
    1183: 40,
    139: 40,
    1184: 5,
    1185: 40,
    156: 38,
    408: 40,
    213: 39,
    1186: 40,
    1187: 40,
    1188: 11,
    115: 40,
    1189: 40,
    304: 40,
    1190: 40,
    312: 40,
    233: 39,
    286: 40,
    264: 40,
    1191: 4,
    356: 40,
    25: 39,
    750: 40,
    269: 40,
    307: 39,
    410: 39,
    730: 38,
    216: 40,
    1192: 38,
    119: 40,
    682: 40,
    434: 40,
    126: 39,
    919: 40,
    85: 39,
    1193: 7,
    108: 7,
    135: 40,
    1194: 40,
    432: 40,
    53: 40,
    1195: 40,
    111: 40,
    305: 38,
    1125: 40,
    1196: 40,
    1197: 21,
    1198: 40,
    1199: 40,
    1200: 40,
    378: 40,
    591: 40,
    92: 40,
    1098: 40,
    291: 40,
    1063: 38,
    1135: 40,
    189: 40,
    245: 40,
    194: 40,
    1201: 38,
    386: 40,
    1202: 39,
    857: 40,
    452: 40,
    1203: 40,
    346: 40,
    152: 38,
    83: 40,
    1204: 1,
    726: 40,
    61: 40,
    39: 18,
    1117: 39,
    1205: 40,
    415: 40,
    1206: 40,
    153: 39,
    1207: 40,
    129: 39,
    220: 40,
    1208: 8,
    231: 40,
    1209: 39,
    1210: 40,
    117: 38,
    822: 39,
    238: 40,
    143: 39,
    1211: 40,
    228: 40,
    494: 4,
    226: 40,
    91: 39,
    1072: 37,
    435: 40,
    345: 40,
    893: 40,
    621: 40,
    1212: 40,
    297: 40,
    1213: 23,
    1214: 40,
    1215: 38,
    529: 40,
    1216: 38,
    1217: 40,
    1218: 11,
    1219: 38,
    1220: 38,
    525: 39,
    204: 40,
    693: 40,
    179: 35,
    1221: 40,
    1222: 40,
    1223: 40,
    1224: 40,
    1225: 22,
    1226: 40,
    1227: 39,
    571: 40,
    1228: 40,
    556: 40,
    280: 40,
    1229: 40,
    1230: 37,
    1231: 40,
    1232: 37,
    746: 40,
    1233: 40,
    1234: 40,
    144: 40,
    282: 39,
    167: 40,
    1235: 40,
    1236: 40,
    1237: 40,
    234: 39,
    563: 40,
    1238: 37,
    1239: 40,
    1240: 40,
    366: 40,
    816: 40,
    1241: 40,
    719: 40,
    284: 40,
    1242: 39,
    247: 40,
    1243: 1,
    1244: 39,
    1245: 29,
    1246: 40,
    1247: 40,
    592: 40,
    385: 3,
    1248: 40,
    1249: 40,
    133: 40,
    301: 38,
    1250: 40,
    379: 38,
    1251: 40,
    450: 40,
    1252: 37,
    316: 40,
    1253: 29,
    1254: 31,
    461: 40,
    1255: 40,
    1256: 39,
    599: 40,
    281: 40,
    1257: 33,
    1258: 40,
    1259: 40,
    319: 40,
    1260: 40,
    1261: 40,
    546: 40,
    1262: 40,
    1263: 40,
    1264: 37,
    1265: 40,
    1266: 40,
    1267: 20,
    1268: 40,
    1269: 40,
    689: 40,
    1270: 39,
    1271: 29,
    1272: 40,
    354: 39,
    339: 40,
    1009: 40,
    1273: 40,
    1274: 40,
    1275: 40,
    361: 40,
    1276: 40,
    326: 39,
    1277: 40,
    1278: 40,
    1279: 40,
    212: 40,
    1280: 40,
    1281: 40,
    794: 40,
    1282: 40,
    955: 40,
    387: 40,
    523: 40,
    389: 39,
    1283: 15,
    146: 38,
    372: 40,
    289: 39,
    440: 37,
    321: 40,
    976: 38,
    1284: 40,
    1285: 40,
    357: 27,
    1286: 40,
    1287: 40,
    365: 40,
    1288: 37,
    81: 39,
    1289: 40,
    1290: 39,
    948: 40,
    174: 40,
    1028: 40,
    1291: 5,
    1292: 40,
    1005: 40,
    235: 38,
    1293: 40,
    1294: 40,
    1295: 38,
    1296: 40,
    1297: 37,
    1298: 40,
    1299: 29,
    1300: 40,
    1301: 21,
    1051: 40,
    566: 39,
    1302: 40,
    1062: 24,
    1303: 21,
    1304: 40,
    1305: 40,
    1306: 40,
    298: 40,
    1307: 40,
    1308: 40,
    1309: 40,
    43: 39,
    1310: 38,
    593: 40,
    1311: 40,
    1312: 40,
    749: 35,
    623: 40,
    1313: 6,
    265: 40,
    1314: 40,
    1315: 40,
    448: 38,
    257: 40,
    1316: 15,
    786: 4,
    801: 40,
    972: 40,
    1317: 40,
    1318: 40,
    657: 29,
    561: 40,
    513: 38,
    411: 39,
    1122: 38,
    922: 40,
    518: 40,
    814: 40,
    1319: 40,
    1320: 40,
    649: 8,
    607: 40,
    819: 40,
    1321: 40,
    1322: 3,
    227: 40,
    817: 40,
    712: 40,
    1323: 40,
    1324: 40,
    673: 29,
    459: 40,
    643: 40,
    765: 39,
    1008: 40,
    225: 40,
    1083: 40,
    813: 40,
    1145: 35,
    796: 40,
    1325: 40,
    363: 39,
    1326: 40,
    997: 40,
    1327: 40,
    1328: 40,
    1329: 40,
    182: 40,
    1330: 40,
    1331: 40,
    1332: 40,
    1333: 40,
    939: 40,
    1334: 40,
    480: 37,
    907: 40,
    1335: 15,
    1336: 40,
    829: 40,
    947: 1,
    1116: 40,
    733: 40,
    123: 40,
    506: 37,
    569: 8,
    1337: 40,
    1338: 5,
    1339: 40,
    1340: 38,
    851: 39,
    142: 40,
    436: 40,
    1341: 39,
    1342: 21,
    885: 5,
    815: 3,
    401: 40,
    1343: 40,
    1344: 40,
    1345: 8,
    160: 38,
    1126: 40,
    1346: 40,
    332: 40,
    397: 40,
    551: 40,
    1347: 2,
    1348: 40,
    803: 40,
    484: 39,
    1349: 4,
    1350: 40,
    222: 7,
    1351: 39,
    1352: 40,
    828: 40,
    1353: 40,
    612: 40,
    1354: 40,
    1355: 7,
    1356: 37,
    1357: 40,
}


if __name__ == "__main__":
    args = _parse_args()
    _main(
        args.scans_root_dir_path,
        args.remove_semantic_and_instance,
        args.compress,
        args.jobs,
    )
