#!/bin/sh

if [ ! -d /tmp/$USER/data_cub ]; then
    echo "Copying dataset to local /tmp..."
    mkdir -p /tmp/$USER/data_cub
    rsync -ah --info=progress2 --ignore-existing ./data/ /tmp/$USER/data_cub
    echo "Dataset copied to local /tmp."
else
    echo "Using cached local dataset."
fi