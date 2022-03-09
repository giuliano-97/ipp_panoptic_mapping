#!/bin/bash

if [ -z $1 ]
then
    echo "Usage prepare_scan.sh <PATH_TO_SCAN>"
fi

# Convert to absolute path
SCAN_DIR=$(realpath $1)
if [ ! -d $SCAN_DIR ]
then
    echo "$SCAN_DIR is not a valid scan dir path!"
fi

SCAN_ID=$(basename $SCAN_DIR)

# Move to repo root
cd "$(dirname "$(dirname "${BASH_SOURCE[0]}")" )" >/dev/null
export PYTHONPATH=.

# Create gt panoptic labeled pointcloud
if [ ! -f $SCAN_DIR/${SCAN_ID}.pointcloud.ply ]
then
    echo "Creating panoptic labeled ground truth pointcloud"
    python3 tools/create_pano_pcd_from_scan_gt.py $SCAN_DIR
fi

if [ ! -f $SCAN_DIR/${SCAN_ID}_labels.csv ]
then
    echo "Creating labels csv file"
    python3 tools/create_pano_pcd_gt_labels_csv.py $SCAN_DIR/${SCAN_ID}.pointcloud.ply
fi

# Create groundtruth panoptic maps
if [ ! -d $SCAN_DIR/panoptic ]
then
    echo "Creating ground truth temporally consistent panoptic maps"
    python3 tools/create_temporally_consistent_panoptic_maps.py $SCAN_DIR
fi

# Create detectron style groundtruth labels
if [ ! -d $SCAN_DIR/panoptic_ground_truth ]
then
    echo "Creating detectron style ground truth labels"
    python3 tools/create_detectron_pano_seg_labels.py $SCAN_DIR/panoptic $SCAN_DIR/panoptic_ground_truth
fi