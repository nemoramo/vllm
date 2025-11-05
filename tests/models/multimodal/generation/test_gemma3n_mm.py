import torch

from vllm.model_executor.models.gemma3n_mm import Gemma3nForConditionalGeneration


def test_collate_audio_batch_varlen_list():
    batch = [
        torch.ones(1, 2, 3),
        2 * torch.ones(1, 4, 3),
    ]

    collated = Gemma3nForConditionalGeneration._collate_audio_batch(
        batch, pad_value=0.0
    )

    assert isinstance(collated, torch.Tensor)
    assert collated.shape == (2, 1, 4, 3)
    assert torch.all(collated[0, :, :2] == 1)
    assert torch.all(collated[0, :, 2:] == 0)
    assert torch.all(collated[1, :, :4] == 2)


def test_collate_audio_batch_mask_padding():
    masks = [
        torch.tensor([[False, False, True]]),
        torch.tensor([[False, True, True, True]]),
    ]

    collated = Gemma3nForConditionalGeneration._collate_audio_batch(
        masks, pad_value=True
    )

    assert collated.dtype == torch.bool
    assert collated.shape == (2, 1, 4)
    assert not collated[0, 0, 0]
    assert collated[0, 0, 2]
    assert collated[0, 0, 3]
