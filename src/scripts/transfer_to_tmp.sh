#!/bin/sh

if [ ! -d /tmp/$USER/data ]; then
    echo "Copying dataset to local /tmp..."
    mkdir -p /tmp/$USER/data
    rsync -ah --info=progress2 --ignore-existing ./data/ /tmp/$USER/data
    echo "Dataset copied to local /tmp."
else
    echo "Using cached local dataset."
fi