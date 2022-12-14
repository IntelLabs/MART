
.PHONY: help
help:  ## Show help
	@grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: clean
clean: ## Clean autogenerated files
	rm -rf dist
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -f .coverage

.PHONY: clean-logs
clean-logs: ## Clean logs
	rm -r logs/**

.PHONY: style
style: ## Run pre-commit hooks
	pre-commit run -a

.PHONY: sync
sync: ## Merge changes from main branch to your current branch
	git fetch --all
	git merge main

.PHONY: test
test: ## Run not slow tests
	pytest -k "not slow"

.PHONY: test-full
test-full: ## Run all tests
	pytest

.PHONY: debug
debug: ## Enter debugging mode with pdb, an example.
	#
	# tips:
	# - use "breakpoint()" to set breakpoint
	# - use "h" to print all commands
	# - use "n" to execute the next line
	# - use "c" to run until the breakpoint is hit
	# - use "l" to print src code around current line, "ll" for full function code
	# - docs: https://docs.python.org/3/library/pdb.html
	#
	python -m pdb -m mart experiment=CIFAR10_CNN debug=default

.PHONY: cifar_attack
cifar_attack: ## Evaluate adversarial robustness of a CIFAR-10 model from robustbench.
	python -m mart experiment=CIFAR10_RobustBench \
	trainer=gpu \
	fit=false \
	+trainer.limit_test_batches=1 \
	+attack@model.modules.input_adv_test=classification_eps8_pgd10_step1

.PHONY: cifar_train
cifar_train: ## Adversarial training for a CIFAR-10 model.
	python -m mart experiment=CIFAR10_CNN_Adv \
	fit=true \
	trainer=gpu


# Download and extract dataset of carla_over_obj_det
CARLA_OVERHEAD_DATASET_TRAIN ?= data/carla_over_obj_det/train/kwcoco_annotations.json
CARLA_OVERHEAD_DATASET_DEV ?= data/carla_over_obj_det/dev/kwcoco_annotations.json

data/carla_over_obj_det/carla_over_od_dev_1.0.0.tar.gz:
	mkdir -p $(@D)
	wget -O $@ https://armory-public-data.s3.us-east-2.amazonaws.com/carla/carla_over_od_dev_1.0.0.tar.gz

$(CARLA_OVERHEAD_DATASET_DEV): data/carla_over_obj_det/carla_over_od_dev_1.0.0.tar.gz
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
	tags=["MaskPGD50_LR5"]
