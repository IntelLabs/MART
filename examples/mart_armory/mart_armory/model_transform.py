import torch

from mart.models.dual_mode import DualModeGeneralizedRCNN


class ArtRcnnModelWrapper(torch.nn.Module):
    """Modify the model so that it is convenient to attack.

    Common issues:
        1. Make the model accept a single argument `output=model(batch)`;
        2. Make the model return loss in eval mode;
        3. Change non-differentiable operations.
    """

    def __init__(self, model):
        super().__init__()

        # Extract PyTorch model from an ART Estimator.
        # TODO: Automatically search for torch.nn.Module within model.
        self.model = DualModeGeneralizedRCNN(model._model)

    def forward(self, batch):
        # Make the model accept batch as an argument parameter.
        output = self.model(*batch)
        return output


class ListInputAsArgs:
    def __call__(self, model):
        def forward(batch):
            return model(*batch)

        return forward


class Extract:
    def __init__(self, attrib):
        self.attrib = attrib

    def __call__(self, model):
        model = getattr(model, self.attrib)
        return model
