#!/bin/bash

cd $(dirname $0)
source common.sh

export REGION=us-central1
export JOB_NAME=hair
export JOB_ID=${JOB_NAME}_$(date +%Y%m%d_%H%M%S)

MACHINE_TYPE=n1-standard-4
#MACHINE_TYPE=standard_p100

GPU_COUNT=1
GPU_TYPE=NVIDIA_TESLA_P100

EPOCHS=30
BATCHSIZE=16
DATA=gs://hairy/full256_0121.tar
MODELSAVEDIR=gs://hairy/models/leakyrelu1_55555_p100_b16

OLDMODEL=gs://hairy/models/leakyrelu1_55555_p100/20210228-2226/epoch_010.bson

# 	   --master-machine-type=n1-standard-8 \
	#	   --master-accelerator=count=${GPU_COUNT},type=${GPU_TYPE} \


gcloud ai-platform jobs submit training "$JOB_ID" \
	   --region="$REGION" \
	   --master-image-uri="$IMAGE_URI" \
	   --scale-tier=CUSTOM \
 	   --master-machine-type=$MACHINE_TYPE \
	   --master-accelerator=count=${GPU_COUNT},type=${GPU_TYPE} \
	   -- \
	   scripts/train.jl \
	   --data="$DATA" \
	   --modelsavedir="$MODELSAVEDIR" \
	   --epochs=$EPOCHS \
	   --batch_size=$BATCHSIZE \
	   --previous_model=$OLDMODEL
