# Adversarial Robustness of Anomaly Detection

This project demonstrates how to generate adversarial examples against anomaly detection models in [Anomalib](https://github.com/openvinotoolkit/anomalib).

## Installation

Anomalib requires Python 3.10+.

```sh
pip install -r requirements.txt
```

## Experiment

0. \[Optional\] Soft link the existing datasets folder from Anomalib if you have downloaded datasets before with Anomalib.

```sh
ln -s {PATH_TO_ANOMALIB_REPO}/datasets .
```

1. Train a model. The config file [configs/anomalib/stfpm.yaml](configs/anomalib/stfpm.yaml) adds an EarlyStopping Callback with maximal 100 epochs.

```sh
CUDA_VISIBLE_DEVICES=0 anomalib train \
--data anomalib.data.MVTec \
--data.category transistor \
--config configs/anomalib/stfpm.yaml
```

2. Evaluate the trained model without adversary as baseline.

```sh
CUDA_VISIBLE_DEVICES=0 anomalib test \
--data anomalib.data.MVTec \
--data.category transistor \
--config configs/anomalib/stfpm.yaml \
--ckpt_path=results/Stfpm/MVTec/transistor/latest/weights/lightning/model.ckpt
```

```console
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│        image_AUROC        │    0.8733333349227905     │
│       image_F1Score       │    0.7945205569267273     │
│        pixel_AUROC        │    0.7860202789306641     │
└───────────────────────────┴───────────────────────────┘
```

2. Generate an adversary config from MART.

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

3. Run attack. The config file [configs/anomalib/stfpm_mart.yaml](configs/anomalib/stfpm_mart.yaml) adds a MART callback that loads the attack config file we just generated [./anomalib_fgsm_linf_10.yaml](./anomalib_fgsm_linf_10.yaml).

```sh
CUDA_VISIBLE_DEVICES=0 anomalib test \
--data anomalib.data.MVTec \
--data.category transistor \
--config configs/anomalib/stfpm_mart.yaml \
--ckpt_path=results/Stfpm/MVTec/transistor/latest/weights/lightning/model.ckpt
```

```console
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃        Test metric        ┃       DataLoader 0        ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│        image_AUROC        │    0.5979167222976685     │
│       image_F1Score       │    0.5714285969734192     │
│        pixel_AUROC        │    0.6867808699607849     │
└───────────────────────────┴───────────────────────────┘
```
