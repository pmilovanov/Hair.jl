#!/bin/bash

docker run --rm -it --gpus all \
	 -v ~/hair:/app \
	   -v ~/.config/gcloud:/root/.config/gcloud \
	   hair0 "$@"
