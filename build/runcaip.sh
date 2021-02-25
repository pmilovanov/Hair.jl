#!/bin/bash

cd $(dirname $0)
source common.sh

export REGION=us-central1
export JOB_NAME=hair
export JOB_ID=${JOB_NAME}_$(date +%Y%m%d_%H%M%S)

MACHINE_TYPE=n1-standard-8

GPU_COUNT=1
GPU_TYPE=NVIDIA_TESLA_K80

DATA=gs://hairy/full256_0121.tar
MODELSAVEDIR=gs://hairy/models/leakyrelu1_55555

# 	   --master-machine-type=n1-standard-8 \
#	   --master-accelerator=count=${GPU_COUNT},type=${GPU_TYPE} \


gcloud ai-platform jobs submit training "$JOB_ID" \
	   --region="$REGION" \
	   --master-image-uri="$IMAGE_URI" \
	   --scale-tier=BASIC_GPU \
	   -- \
	   scripts/train.jl \
	   --data="$DATA" \
	   --modelsavedir="$MODELSAVEDIR"
