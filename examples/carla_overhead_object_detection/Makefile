# Download and extract dataset of carla_over_obj_det
CARLA_OVERHEAD_DATASET_TRAIN ?= data/carla_over_obj_det/train/kwcoco_annotations.json
CARLA_OVERHEAD_DATASET_DEV ?= data/carla_over_obj_det/dev/kwcoco_annotations.json

data/carla_over_obj_det/carla_over_od_dev_2.0.0.tar.gz:
	mkdir -p $(@D)
	wget -O $@ https://armory-public-data.s3.us-east-2.amazonaws.com/carla/carla_over_od_dev_2.0.0.tar.gz

$(CARLA_OVERHEAD_DATASET_DEV): data/carla_over_obj_det/carla_over_od_dev_2.0.0.tar.gz
	tar -zxf $<  -C data/carla_over_obj_det

data/carla_over_obj_det/carla_over_od_train_val_1.0.0.tar.gz:
	mkdir -p $(@D)
	wget -O $@ https://armory-public-data.s3.us-east-2.amazonaws.com/carla/carla_over_od_train_val_1.0.0.tar.gz

$(CARLA_OVERHEAD_DATASET_TRAIN): data/carla_over_obj_det/carla_over_od_train_val_1.0.0.tar.gz
	tar -zxf $<  -C data/carla_over_obj_det


.PHONY: carla_train
carla_train: $(CARLA_OVERHEAD_DATASET_TRAIN) $(CARLA_OVERHEAD_DATASET_DEV) ## Train Faster R-CNN with the CarlaOverObjDet dataset from Armory.
	python -m mart \
	experiment=ArmoryCarlaOverObjDet_TorchvisionFasterRCNN \
	trainer=gpu \
	trainer.precision=16 \
	fit=true \
	tags=["regular_training","backbone_ImageNetPretrained"] \


# You need to specify weights of target model in [model.modules.losses_and_detections.model.weights_fpath].
.PHONY: carla_attack
carla_attack: $(CARLA_OVERHEAD_DATASET_TRAIN) $(CARLA_OVERHEAD_DATASET_DEV) ## Evaluate adversarial robustness of a pretrained model.
	python -m mart \
	experiment=ArmoryCarlaOverObjDet_TorchvisionFasterRCNN \
	trainer=gpu \
	fit=false \
	model.modules.losses_and_detections.model.weights_fpath=null \
	+attack@model.modules.input_adv_test=object_detection_mask_adversary \
	model.modules.input_adv_test.optimizer.lr=5 \
	model.modules.input_adv_test.max_iters=50 \
	+model.test_sequence.seq001.input_adv_test._call_with_args_=[input,target] \
	+model.test_sequence.seq001.input_adv_test.model=model \
	+model.test_sequence.seq001.input_adv_test.step=step \
	model.test_sequence.seq010.preprocessor=[input_adv_test] \
	# tags=["MaskPGD50_LR5"]
