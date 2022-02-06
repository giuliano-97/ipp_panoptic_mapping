#!/usr/bin/bash


if [ ! -d $1 ]
then
    echo "Usage create_voxel_segs.sh <PANOPTIC_CLOUDS_DIR_PATH>"
    exit 1
fi

CLOUD_TO_SEGS=$(dirname $(dirname "${BASH_SOURCE[0]}"))/build/panoptic_mapping_evaluation/cloud_to_segs
if [ ! -f $CLOUD_TO_SEGS ]
then
    echo "cloud_to_segs binary not found - did you build the code correctly?"
    exit 1
fi

PANOPTIC_CLOUDS_DIR_PATH=$1
PANOPTIC_CLOUD_FILES=$(find ${PANOPTIC_CLOUDS_DIR_PATH} -name \*.pointcloud.ply)

for FILE in $PANOPTIC_CLOUD_FILES
do
    echo "Processing file $(basename $FILE)..."
    $CLOUD_TO_SEGS $FILE
    echo "Done."
done