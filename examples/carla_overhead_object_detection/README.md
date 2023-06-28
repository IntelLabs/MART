# Introduction

This example shows how to use MART to train an object detection model on the Carla overhead dataset.

## Installation

```bash
pip install -r requirements.txt
```

```bash
# train on 1 GPU
CUDA_VISIBLE_DEVICES=0 \
python -m mart experiment=ArmoryCarlaOverObjDet_TorchvisionFasterRCNN \
	task_name=1GPU_ArmoryCarlaOverObjDet_TorchvisionFasterRCNN \
	trainer=gpu \
	fit=true

# train on multiple GPUs using Distributed Data Parallel
CUDA_VISIBLE_DEVICES=0,1 \
python -m mart experiment=ArmoryCarlaOverObjDet_TorchvisionFasterRCNN \
	task_name=2GPUs_ArmoryCarlaOverObjDet_TorchvisionFasterRCNN \
	fit=true \
	trainer=ddp \
	trainer.devices=2 \
	datamodule.ims_per_batch=4 \
	model.optimizer.lr=0.025 \
	trainer.max_steps=5244
```
