import copy

import numpy as np
import torch

from soil_learning.config import ExperimentConfig
from soil_learning.data import Examples
from soil_learning.federated import fedavg_round, fedper_round, predict, weighted_average
from soil_learning.models import build_model

CONFIG = ExperimentConfig(batch_size=8)
TRAIN = {"seed": 42, "lr": 1e-2, "epochs": 1}


def examples(n: int, seed: int) -> Examples:
    rng = np.random.default_rng(seed)
    t0 = (np.datetime64("2017-01-01") + np.arange(n).astype("timedelta64[D]")).astype("datetime64[ns]")
    return Examples(
        dynamic=rng.random((n, 30, 6), dtype=np.float32), static=np.zeros((n, 4), dtype=np.float32),
        y=(0.1 + 0.4 * rng.random(n)).astype(np.float32), persistence=rng.random(n, dtype=np.float32),
        t0=t0, target_date=t0 + np.timedelta64(14, "D"),
    )


def equal(a, b) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def setup(architecture: str = "lstm"):
    torch.manual_seed(0)
    factory = lambda: build_model(architecture)
    return factory, factory()


def test_weighted_average_uses_example_counts():
    a, b = {"w": torch.tensor([0.0, 2.0])}, {"w": torch.tensor([4.0, 6.0])}
    assert torch.allclose(weighted_average([(a, 1), (b, 3)])["w"], torch.tensor([3.0, 5.0]))


def test_fedavg_aggregates_the_head():
    factory, model = setup("tcn")
    data = {"A": examples(32, 1), "B": examples(32, 2)}
    new_state, _ = fedavg_round(factory, model.state_dict(), data, CONFIG, round_idx=1, **TRAIN)
    assert new_state.keys() == model.state_dict().keys()
    assert not torch.equal(new_state["head.3.weight"], model.state_dict()["head.3.weight"])


def test_fedper_head_stays_on_the_client():
    factory, model = setup()
    heads = {}
    data = {"A": examples(32, 1), "B": examples(32, 2)}
    shared, _ = fedper_round(factory, model.shared_state_dict(), heads, model.private_state_dict(), data,
                             CONFIG, round_idx=1, **TRAIN)
    assert not any(key.startswith("head.") for key in shared)
    assert set(heads) == {"A", "B"}
    assert not equal(heads["A"], heads["B"])
    assert not equal(heads["A"], model.private_state_dict())


def test_fedper_head_persists_when_client_does_not_train():
    factory, model = setup()
    heads = {}
    data = {"A": examples(32, 1), "B": examples(32, 2)}
    shared, _ = fedper_round(factory, model.shared_state_dict(), heads, model.private_state_dict(), data,
                             CONFIG, round_idx=1, **TRAIN)
    before = copy.deepcopy(heads)
    fedper_round(factory, shared, heads, model.private_state_dict(), {"A": examples(40, 3)},
                 CONFIG, round_idx=2, **TRAIN)
    assert equal(heads["B"], before["B"])
    assert not equal(heads["A"], before["A"])


def test_zero_residual_reproduces_persistence():
    _, model = setup()
    with torch.no_grad():
        model.head[-1].weight.zero_(); model.head[-1].bias.zero_()
    data = examples(16, 5)
    assert np.array_equal(predict(model, data, CONFIG), data.persistence)
