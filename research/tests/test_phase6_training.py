import numpy as np
import pytest

from research.pipeline.phase6_training import (
    SparsePlasticReadout,
    DecoderMLP,
    prune_edge_budget,
    predict_model,
    purged_time_split,
    select_hyperparameters_train_only,
    spearman_ic,
)


def test_train_internal_split_purges_sixty_bars():
    train_idx, inner_val_idx = purged_time_split(
        n_samples=100,
        sample_stride_bars=6,
        purge_bars=60,
        train_fraction=0.8,
    )

    assert train_idx.tolist() == list(range(80))
    assert inner_val_idx.tolist() == list(range(90, 100))
    assert (inner_val_idx[0] - train_idx[-1]) * 6 >= 60


def test_edge_budget_pruning_is_deterministic_and_drops_smallest_magnitudes():
    edges = (
        np.array([0, 0, 1, 1, 2]),
        np.array([4, 3, 5, 2, 1]),
        np.array([0.1, -0.8, 0.6, 0.2, -0.4]),
    )

    first = prune_edge_budget(*edges, budget=3)
    second = prune_edge_budget(*edges, budget=3)

    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(first[0], [0, 1, 2])
    np.testing.assert_array_equal(first[1], [3, 5, 1])
    np.testing.assert_allclose(first[2], [-0.8, 0.6, -0.4])


def test_sparse_readout_mask_blocks_gradients_and_updates_outside_edges():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU unavailable")
    device = torch.device("cuda")
    source_indices = torch.tensor([[0, 1, 0], [1, 0, 0]], device=device)
    base_weights = torch.tensor([[0.4, -0.2, 0.0], [0.3, 0.0, 0.0]], device=device)
    active_mask = torch.tensor([[True, True, False], [True, False, False]], device=device)
    model = SparsePlasticReadout(source_indices, base_weights, active_mask).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.1)
    x = torch.randn(8, 2, device=device)
    before = model.delta.detach().clone()

    pred_return, pred_action = model(x)
    (pred_return.square().mean() + pred_action.square().mean()).backward()

    assert torch.equal(model.delta.grad[~active_mask], torch.zeros_like(model.delta.grad[~active_mask]))
    optimizer.step()
    assert torch.equal(model.delta.detach()[~active_mask], before[~active_mask])


def test_hyperparameter_selector_rejects_validation_inputs():
    x = np.ones((20, 4), dtype=np.float32)
    y_cont = np.linspace(-1, 1, 20, dtype=np.float32)
    y_action = np.array(["BUY", "HOLD", "SELL", "BUY", "HOLD"] * 4)

    with pytest.raises(TypeError):
        select_hyperparameters_train_only(
            x,
            y_cont,
            y_action,
            sample_stride_bars=6,
            seed=0,
            val_features=x,
        )


def test_known_signal_has_positive_ic_and_pure_noise_is_near_zero():
    rng = np.random.default_rng(20260920)
    target = rng.normal(size=5000)
    signal = target + rng.normal(scale=0.5, size=len(target))
    noise = rng.normal(size=len(target))

    assert spearman_ic(target, signal) > 0.5
    assert abs(spearman_ic(target, noise)) < 0.05


def test_predict_model_returns_continuous_predictions_and_three_action_logits():
    torch = pytest.importorskip("torch")
    model = DecoderMLP(input_dim=2, hidden_dim=64, dropout=0.1).eval()
    x = np.array([[0.0, 1.0], [1.0, 0.0], [-1.0, 2.0]], dtype=np.float32)
    normalization = {
        "x_mean": [0.0, 0.0],
        "x_std": [1.0, 1.0],
        "target_mean": 0.25,
        "target_std": 2.0,
    }

    pred_return, action_logits = predict_model(
        model, x, normalization, batch_size=2, device="cpu"
    )

    assert pred_return.shape == (3,)
    assert action_logits.shape == (3, 3)
    assert np.isfinite(pred_return).all()
    assert np.isfinite(action_logits).all()


def test_phase6a_decoder_parameter_budget_is_explicit_and_equal_across_variants():
    model = DecoderMLP(input_dim=1291, hidden_dim=64, dropout=0.1)

    assert sum(parameter.numel() for parameter in model.parameters()) == 82948


def test_torch_reservoir_can_return_time_mean_for_selected_neurons():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU unavailable")
    import scipy.sparse as sp

    from research.pipeline.torch_sim import TorchReservoir
    from src.connectome.schema import ConnectomeGraph

    graph = ConnectomeGraph(
        weights=sp.csr_matrix((3, 3), dtype=np.float32),
        neuron_ids=["s", "m0", "m1"],
        neuron_types=["sensory", "motor", "motor"],
        sensory_idx=np.array([0], dtype=np.int64),
        motor_idx=np.array([1, 2], dtype=np.int64),
    )
    reservoir = TorchReservoir(graph)
    currents = np.ones((2, 4, 1), dtype=np.float32)

    motor, recorded_mean = reservoir.simulate_batch(
        currents,
        gain=1.0,
        leak=0.5,
        record_activity_indices=np.array([0, 1], dtype=np.int64),
    )

    state = 0.0
    trace = []
    for _ in range(4):
        state = 0.5 * state + 0.5 * np.tanh(1.0)
        trace.append(state)
    np.testing.assert_allclose(recorded_mean[:, 0], np.mean(trace), rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(recorded_mean[:, 1], np.zeros(2, dtype=np.float32))
    assert motor.shape == (2, 4, 2)
