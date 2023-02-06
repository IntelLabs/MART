from typing import Any, Dict, List, Tuple

import hydra
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase

from mart import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def lightning(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Contains training pipeline. Instantiates all PyTorch Lightning objects from config.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # Set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    # Init lightning datamodule
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    # Init lightning model
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # Init lightning callbacks
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    # Init lightning loggers
    logger: List[LightningLoggerBase] = utils.instantiate_loggers(cfg.get("logger"))

    # Init lightning trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
    )

    # Send some parameters from config to all lightning loggers
    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    # ckpt_path could be None if resume=null.
    ckpt_path = cfg.get("ckpt_path", None)

    # Train the model
    if cfg.get("fit"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        ckpt_path = None  # make sure trainer tests trained model

    train_metrics = trainer.callback_metrics

    # Evaluate model on test set, using the best model achieved during training
    if cfg.get("test"):
        log.info("Starting testing!")
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    # Print path to best checkpoint
    if not cfg.trainer.get("fast_dev_run"):
        log.info(f"Best model ckpt: {trainer.checkpoint_callback.best_model_path}")

    return metric_dict, object_dict
