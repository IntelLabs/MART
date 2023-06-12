# What does this PR do?

<!--
Please include a summary of the change and which issue is fixed.
Please also include relevant motivation and context.
List any dependencies that are required for this change.
List all the breaking changes introduced by this pull request.
-->

Fixes #\<issue_number>

## Type of change

Please check all relevant options.

- [ ] Improvement (non-breaking)
- [ ] Bug fix (non-breaking)
- [ ] New feature (non-breaking)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] This change requires a documentation update

## Testing

Please describe the tests that you ran to verify your changes. Consider listing any relevant details of your test configuration.

- [ ] `pytest`
- [ ] `CUDA_VISIBLE_DEVICES=0 python -m mart experiment=CIFAR10_CNN_Adv trainer=gpu trainer.precision=16` reports 70% (21 sec/epoch).
- [ ] `CUDA_VISIBLE_DEVICES=0,1 python -m mart experiment=CIFAR10_CNN_Adv trainer=ddp trainer.precision=16 trainer.devices=2 model.optimizer.lr=0.2 trainer.max_steps=2925 datamodule.ims_per_batch=256 datamodule.world_size=2` reports 70% (14 sec/epoch).

## Before submitting

- [ ] The title is **self-explanatory** and the description **concisely** explains the PR
- [ ] My **PR does only one thing**, instead of bundling different changes together
- [ ] I list all the **breaking changes** introduced by this pull request
- [ ] I have commented my code
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] I have run pre-commit hooks with `pre-commit run -a` command without errors

## Did you have fun?

Make sure you had fun coding 🙃
