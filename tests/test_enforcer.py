#
# Copyright (C) 2022 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#

import pytest
import torch

from mart.attack.enforcer import ConstraintViolated, Enforcer, Integer, Lp, Mask, Range


def test_constraint_range():
    input = torch.tensor([0, 0, 0])
    target = None

    constraint = Range(min=0, max=255)

    perturbation = torch.tensor([0, 128, 255])
    constraint(input + perturbation, input=input, target=target)

    with pytest.raises(ConstraintViolated):
        perturbation = torch.tensor([0, -1, 255])
        constraint(input + perturbation, input=input, target=target)
        perturbation = torch.tensor([0, 1, 256])
        constraint(input + perturbation, input=input, target=target)


def test_constraint_l2():
    input = torch.zeros((3, 10, 10))
    batch_input = torch.stack((input, input))

    constraint = Lp(eps=17.33, p=2, dim=[-1, -2, -3])
    target = None

    # (3*10*10)**0.5 = 17.3205
    perturbation = torch.ones((3, 10, 10))
    constraint(input + perturbation, input=input, target=target)
    constraint(batch_input + perturbation, input=batch_input, target=target)

    with pytest.raises(ConstraintViolated):
        constraint(batch_input + perturbation * 2, input=input, target=target)
        constraint(batch_input + perturbation * 2, input=batch_input, target=target)


def test_constraint_integer():
    input, target = None, None

    constraint = Integer()

    input_adv = torch.tensor([1.0, 2.0])
    constraint(input_adv, input=input, target=target)

    input_adv = torch.tensor([1.0, 2.001])
    with pytest.raises(ConstraintViolated):
        constraint(input_adv, input=input, target=target)


def test_constraint_mask():
    input = torch.zeros((3, 2, 2))
    perturbation = torch.ones((3, 2, 2))
    mask = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    target = {"perturbable_mask": mask}

    constraint = Mask()

    constraint(input + perturbation * mask, input=input, target=target)
    with pytest.raises(ConstraintViolated):
        constraint(input + perturbation, input=input, target=target)


def test_enforcer_non_modality():
    enforcer = Enforcer(constraints={"range": Range(min=0, max=255)})

    input = torch.tensor([0, 0, 0])
    perturbation = torch.tensor([0, 128, 255])
    input_adv = input + perturbation
    target = None

    # tensor input.
    enforcer(input_adv, input=input, target=target)
    # list of tensor input.
    enforcer([input_adv], input=[input], target=[target])
    # tuple of tensor input.
    enforcer((input_adv,), input=(input,), target=(target,))

    perturbation = torch.tensor([0, -1, 255])
    input_adv = input + perturbation

    with pytest.raises(ConstraintViolated):
        enforcer(input_adv, input=input, target=target)

    with pytest.raises(ConstraintViolated):
        enforcer([input_adv], input=[input], target=[target])

    with pytest.raises(ConstraintViolated):
        enforcer((input_adv,), input=(input,), target=(target,))


# def test_enforcer_modality():
#    # Assume a rgb modality.
#    enforcer = Enforcer(rgb={"range": Range(min=0, max=255)})
#
#    input = torch.tensor([0, 0, 0])
#    perturbation = torch.tensor([0, 128, 255])
#    input_adv = input + perturbation
#    target = None
#
#    # Dictionary input.
#    enforcer({"rgb": input_adv}, input={"rgb": input}, target=target)
#    # List of dictionary input.
#    enforcer([{"rgb": input_adv}], input=[{"rgb": input}], target=[target])
#    # Tuple of dictionary input.
#    enforcer(({"rgb": input_adv},), input=({"rgb": input},), target=(target,))
#
#    perturbation = torch.tensor([0, -1, 255])
#    input_adv = input + perturbation
#
#    with pytest.raises(ConstraintViolated):
#        enforcer({"rgb": input_adv}, input={"rgb": input}, target=target)
#
#    with pytest.raises(ConstraintViolated):
#        enforcer([{"rgb": input_adv}], input=[{"rgb": input}], target=[target])
#
#    with pytest.raises(ConstraintViolated):
#        enforcer(({"rgb": input_adv},), input=({"rgb": input},), target=(target,))
