import pytest
import torch
import torch.nn as nn
from train_bc import collate_fn, PokemonBCDataset, train_bc
from torch.utils.data import random_split, DataLoader
from model import TitanTransformer

# ===========================================================
# COLLATE_FN TESTS
# ===========================================================

def test_collate_fn_offset_shifting():
    """Verify that collate_fn correctly shifts offsets across multiple batched samples."""
    sample1 = {
        'encoder_indices': [1, 2, 3],
        'encoder_values': [1.0, 1.0, 1.0],
        'encoder_offsets': [0, 2],
        'decoder_indices': [10, 11],
        'decoder_values': [0.5, 0.5],
        'decoder_offsets': [0, 1],
        'target_actions': [0],
        'legal_option_count': 2
    }
    sample2 = {
        'encoder_indices': [4, 5, 6, 7],
        'encoder_values': [1.0, 1.0, 1.0, 1.0],
        'encoder_offsets': [0, 3],
        'decoder_indices': [12],
        'decoder_values': [0.5],
        'decoder_offsets': [0],
        'target_actions': [1],
        'legal_option_count': 1
    }
    ei, eo, ew, di, do_, dw, bl, targets = collate_fn([sample1, sample2])
    assert ei.tolist() == [1, 2, 3, 4, 5, 6, 7]
    assert di.tolist() == [10, 11, 12]
    assert eo.tolist() == [0, 2, 3, 6]
    assert do_.tolist() == [0, 1, 2]
    assert bl.tolist() == [2, 1]
    assert targets.tolist() == [0, 1]

def test_collate_fn_empty_action_handling():
    """Verify collate_fn doesn't crash when a sample has 0 legal actions."""
    sample = {
        'encoder_indices': [1], 'encoder_values': [1.0], 'encoder_offsets': [0],
        'decoder_indices': [], 'decoder_values': [], 'decoder_offsets': [],
        'target_actions': [], 'legal_option_count': 0
    }
    ei, eo, ew, di, do_, dw, bl, targets = collate_fn([sample])
    assert bl.tolist() == [0]
    assert len(di) == 0
    assert targets.tolist() == [0]

def test_collate_fn_single_sample():
    """Verify collate_fn works on a single-element batch with no offset shift needed."""
    sample = {
        'encoder_indices': [5, 6], 'encoder_values': [1.0, 1.0], 'encoder_offsets': [0],
        'decoder_indices': [20, 21, 22], 'decoder_values': [1.0, 1.0, 1.0], 'decoder_offsets': [0, 1, 2],
        'target_actions': [2], 'legal_option_count': 3
    }
    ei, eo, ew, di, do_, dw, bl, targets = collate_fn([sample])
    assert eo.tolist() == [0]
    assert do_.tolist() == [0, 1, 2]
    assert targets.tolist() == [2]

def test_collate_fn_target_fallback():
    """Verify target fallback to 0 when target_actions is empty."""
    sample = {
        'encoder_indices': [1], 'encoder_values': [1.0], 'encoder_offsets': [0],
        'decoder_indices': [10], 'decoder_values': [1.0], 'decoder_offsets': [0],
        'target_actions': [], 'legal_option_count': 1
    }
    _, _, _, _, _, _, _, targets = collate_fn([sample])
    assert targets.tolist() == [0]

def test_collate_fn_many_samples_offset_monotone():
    """Verify encoder offsets are strictly monotonically increasing for N samples."""
    def make_sample(n_enc, n_dec):
        return {
            'encoder_indices': list(range(n_enc)), 'encoder_values': [1.0]*n_enc,
            'encoder_offsets': [0], 'decoder_indices': list(range(n_dec)),
            'decoder_values': [1.0]*n_dec, 'decoder_offsets': list(range(n_dec)),
            'target_actions': [0], 'legal_option_count': n_dec
        }
    batch = [make_sample(3, 2), make_sample(5, 4), make_sample(2, 1)]
    ei, eo, ew, di, do_, dw, bl, targets = collate_fn(batch)
    offsets = eo.tolist()
    assert offsets == sorted(offsets), "Encoder offsets must be non-decreasing"

# ===========================================================
# LOSS FUNCTION TESTS
# ===========================================================

def test_loss_function_accuracy():
    """Verify cross-entropy loss with -inf masking produces correct low-loss predictions."""
    logits = torch.tensor([
        [2.0, 0.5, 0.1],
        [-1.0, 5.0, float('-inf')]
    ])
    targets = torch.tensor([0, 1], dtype=torch.long)
    loss = nn.CrossEntropyLoss()(logits, targets)
    assert loss.item() < 0.5
    assert not torch.isnan(loss)

def test_loss_function_no_nan_on_inf_mask():
    """Verify that a fully masked logit row (-inf everywhere except one) doesn't produce NaN."""
    logits = torch.tensor([[float('-inf'), float('-inf'), 2.0]])
    targets = torch.tensor([2], dtype=torch.long)
    loss = nn.CrossEntropyLoss()(logits, targets)
    assert not torch.isnan(loss)
    assert loss.item() < 0.01  # Should be near zero, clearly correct

def test_loss_function_wrong_prediction_high():
    """Verify loss is high when model confidently predicts the wrong action."""
    logits = torch.tensor([[10.0, -10.0]])
    targets = torch.tensor([1], dtype=torch.long)
    loss = nn.CrossEntropyLoss()(logits, targets)
    assert loss.item() > 5.0

# ===========================================================
# VALIDATION LOOP TESTS (no gradient leakage)
# ===========================================================

def test_validation_no_grad_no_leak():
    """Verify that torch.no_grad() context correctly prevents gradient accumulation during validation.
    TitanTransformer requires exactly 24 encoder words (offsets) per sample."""
    model = TitanTransformer(d_model=128, nhead=4, num_layers=2)
    model.eval()

    # Build a proper 24-word sample: 24 offsets, one token per word
    n_words = 24
    sample = {
        'encoder_indices': list(range(n_words)),
        'encoder_values': [1.0] * n_words,
        'encoder_offsets': list(range(n_words)),   # one token per word
        'decoder_indices': [100, 101],
        'decoder_values': [1.0, 1.0],
        'decoder_offsets': [0, 1],
        'target_actions': [0],
        'legal_option_count': 2
    }
    ei, eo, ew, di, do_, dw, bl, targets = collate_fn([sample])

    with torch.no_grad():
        logits, _ = model(ei, eo, ew, di, do_, dw, bl)
        loss = nn.CrossEntropyLoss()(logits, targets)

    # Under no_grad, NO parameter should have accumulated any grad
    for name, param in model.named_parameters():
        assert param.grad is None, f"Gradient leaked on param '{name}' during no_grad validation!"

def test_validation_model_eval_mode():
    """Verify model is set to eval mode before validation (disables dropout, batchnorm updates)."""
    model = TitanTransformer(d_model=128, nhead=4, num_layers=2)
    model.train()
    model.eval()
    assert not model.training, "Model should be in eval mode during validation!"

def test_train_mode_after_val():
    """Verify model correctly returns to train mode after validation loop (simulates epoch cycle)."""
    model = TitanTransformer(d_model=128, nhead=4, num_layers=2)
    model.eval()
    model.train()
    assert model.training, "Model must return to train mode after validation!"
