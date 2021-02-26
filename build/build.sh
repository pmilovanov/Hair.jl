#!/bin/bash


cd $(dirname $0)
source common.sh

cd ..

docker build -t $IMAGE_URI .
