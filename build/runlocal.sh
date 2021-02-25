#!/bin/bash

cd $(dirname $0)
source common.sh

docker run --rm -it --gpus all \
	 -v ~/hair:/app \
	   -v ~/.config/gcloud:/root/.config/gcloud \
	   ${IMAGE_URI}:latest \
	   "$@"
