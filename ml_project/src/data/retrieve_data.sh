#!/usr/bin/env bash

DATA_ROOT=$PROJECT_ROOT/data

PREFIX_PATH=$1

DOWNLOAD_PATH="$DATA_ROOT/$PREFIX_PATH/external"
UNZIP_TO_PATH="$DATA_ROOT/$PREFIX_PATH/raw"

mkdir -p $DOWNLOAD_PATH
mkdir -p $UNZIP_TO_PATH

cd $DOWNLOAD_PATH
kaggle datasets download -d ronitf/heart-disease-uci
unzip heart-disease-uci.zip -d $UNZIP_TO_PATH
cd -
