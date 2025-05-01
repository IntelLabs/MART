# Adversarial Robustness of Anomaly Detection

This project demonstrates how to generate adversarial examples against anomaly detection models in [Anomalib](https://github.com/openvinotoolkit/anomalib).

## Installation

Anomalib requires Python 3.10+.

```sh
pip install -e .
```

## Experiment

0. [Optional] Soft link the existing datasets folder from Anomalib if you have downloaded datasets before with Anomalib.

```sh
ln -s {PATH_TO_ANOMALIB_REPO}/datasets .
```

### Model: STFPM

1. Train a model. We add an EarlyStopping callback in command line.

```sh
CUDA_VISIBLE_DEVICES=0 anomalib train \
--data anomalib.data.MVTec \
--data.category transistor \
--model Stfpm \
--trainer.callbacks lightning.pytorch.callbacks.EarlyStopping \
--trainer.callbacks.patience 5 \
--trainer.callbacks.monitor pixel_AUROC \
--trainer.callbacks.mode max
```

2. Evaluate the trained model without adversary as baseline.

```sh
CUDA_VISIBLE_DEVICES=0 anomalib test \
--data anomalib.data.MVTec \
--data.category transistor \
--model Stfpm \
--ckpt_path=results/Stfpm/MVTec/transistor/latest/weights/lightning/model.ckpt
```

```console
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│        image_AUROC        │    0.8733333349227905     │
│       image_F1Score       │    0.7945205569267273     │
│        pixel_AUROC        │    0.7860202789306641     │
│       pixel_F1Score       │    0.46384868025779724    │
└───────────────────────────┴───────────────────────────┘
```

3. Generate an adversary config file from MART.

```sh
python -m mart.generate_config \
--config_dir="configs" \
--export_node=callbacks.adversary_connector \
--resolve=True \
callbacks=adversary_connector \
batch_c15n@callbacks.adversary_connector.batch_c15n=dict_imagenet_normalized \
callbacks.adversary_connector.adversary=$\{attack\} \
+attack=classification_fgsm_linf \
~attack.gain \
+attack.gain._target_=mart.nn.Get \
+attack.gain.key=loss \
attack.objective=null \
attack.eps=10 \
attack.callbacks.progress_bar.enable=true \
> anomalib_fgsm_linf_10.yaml
```

4. Run attack. We add a MART callback that loads the attack config file we just generated [./anomalib_fgsm_linf_10.yaml](./anomalib_fgsm_linf_10.yaml).

```sh
CUDA_VISIBLE_DEVICES=0 anomalib test \
--data anomalib.data.MVTec \
--data.category transistor \
--model Stfpm \
--trainer.callbacks mart.utils.CallbackInstantiator \
--trainer.callbacks.cfg_path ./anomalib_fgsm_linf_10.yaml \
--ckpt_path=results/Stfpm/MVTec/transistor/latest/weights/lightning/model.ckpt
```

```console
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│        image_AUROC        │    0.5979167222976685     │
│       image_F1Score       │    0.5714285969734192     │
│        pixel_AUROC        │     0.686780571937561     │
│       pixel_F1Score       │    0.0955422893166542     │
└───────────────────────────┴───────────────────────────┘
```

### Model: WinCLIP

1. Evaluate the pre-trained model without adversary as baseline.

```sh
CUDA_VISIBLE_DEVICES=0 \
anomalib test \
--data anomalib.data.MVTec \
--data.category hazelnut \
--model WinClip \
--data.init_args.image_size [240,240] \
--data.init_args.eval_batch_size 16 \
--metrics.pixel=[F1Score,AUROC]
```

```console
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│        image_AUROC        │    0.9074999690055847     │
│       image_F1Score       │     0.882758617401123     │
│        pixel_AUROC        │    0.9510707855224609     │
│       pixel_F1Score       │    0.37700045108795166    │
└───────────────────────────┴───────────────────────────┘
```

2. Run attack.

```sh
CUDA_VISIBLE_DEVICES=0 \
anomalib test \
--data anomalib.data.MVTec \
--data.category hazelnut \
--model WinClip \
--data.init_args.image_size [240,240] \
--data.init_args.eval_batch_size 16 \
--metrics.pixel=[F1Score,AUROC] \
--trainer.callbacks anomalib_adversary.callbacks.SemanticAdversary \
--trainer.callbacks.seed 2024
```

```console
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│        image_AUROC        │          0.1875           │
│       image_F1Score       │    0.7283236980438232     │
│        pixel_AUROC        │    0.8381223678588867     │
│       pixel_F1Score       │    0.07936933636665344    │
└───────────────────────────┴───────────────────────────┘
```
