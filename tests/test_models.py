import torch

from soil_learning.models import TCNRegressor, build_model, count_parameters


def test_parameter_counts():
    assert count_parameters(build_model("lstm")) == (14920, 545)
    assert count_parameters(build_model("tcn")) == (14989, 545)


def test_output_shape_and_private_boundary():
    dynamic, static = torch.randn(4, 30, 6), torch.randn(4, 4)
    for name in ("lstm", "tcn"):
        model = build_model(name)
        assert model(dynamic, static).shape == (4,)
        assert all(key.startswith("head.") for key in model.private_state_dict())
        assert not any(key.startswith("head.") for key in model.shared_state_dict())


def test_tcn_receptive_field():
    assert TCNRegressor().receptive_field == 29
