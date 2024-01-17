from functools import partial

import pytest
import torch as t
from codebook_circuits.mvp.codebook_act_patching import (
    act_patch_attn_codebook,
    codebook_activation_patcher,
)
from codebook_circuits.mvp.codebook_path_patching import (
    hook_fn_add_activation_to_ctx,
    hook_fn_generic_patch_or_freeze,
    hook_fn_generic_patching_from_context,
    single_path_patch,
)
from codebook_circuits.mvp.data import TravelToCityDataset
from transformer_lens import HookedTransformer


@pytest.fixture(scope="session")
def tiny_model():
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    return HookedTransformer.from_pretrained("EleutherAI/pythia-14m").eval().to(device)


@pytest.fixture(scope="session")
def tiny_dataset_and_model(tiny_model):
    dataset = TravelToCityDataset(n_examples=10)
    orig_tokens = tiny_model.to_tokens(dataset.clean_prompts)
    new_tokens = tiny_model.to_tokens(dataset.corrupted_prompts)
    orig_logits, orig_cache = tiny_model.run_with_cache(dataset.clean_prompts)
    new_logits, new_cache = tiny_model.run_with_cache(dataset.corrupted_prompts)
    return (
        tiny_model,
        orig_tokens,
        new_tokens,
        orig_logits,
        orig_cache,
        new_logits,
        new_cache,
    )


def test_act_patch_attn_codebook(tiny_dataset_and_model):
    (
        tiny_model,
        orig_tokens,
        _,
        _,
        orig_cache,
        _,
        new_cache,
    ) = tiny_dataset_and_model

    activation_1 = "blocks.4.hook_resid_post"
    activation_2 = "blocks.5.hook_resid_post"
    position = None
    tiny_model.add_hook(
        name=activation_2,
        hook=partial(act_patch_attn_codebook, position=position, new_cache=new_cache),
    )
    _, cache = tiny_model.run_with_cache(orig_tokens, return_type=None)
    assert t.equal(cache[activation_1], orig_cache[activation_1])
    assert t.equal(cache[activation_2], new_cache[activation_2])


def test_hook_fn_generic_patching_from_context(tiny_dataset_and_model):
    (
        tiny_model,
        orig_tokens,
        _,
        _,
        orig_cache,
        _,
        _,
    ) = tiny_dataset_and_model
    cache_key = "blocks.5.hook_resid_post"
    tiny_model.hook_dict[cache_key].ctx["zeroes_act"] = t.zeros_like(
        orig_cache[cache_key]
    )
    tiny_model.add_hook(
        name=cache_key,
        hook=partial(hook_fn_generic_patching_from_context, user_def_name="zeroes_act"),
    )
    _, cache = tiny_model.run_with_cache(orig_tokens, return_type=None)
    assert t.equal(cache[cache_key], t.zeros_like(orig_cache[cache_key]))


def test_hook_fn_generic_patch_or_freeze(tiny_dataset_and_model):
    (
        tiny_model,
        orig_tokens,
        _,
        _,
        orig_cache,
        _,
        new_cache,
    ) = tiny_dataset_and_model
    activation_name_1 = "blocks.4.hook_resid_post"
    activation_name_2 = "blocks.5.hook_resid_post"
    position = None
    tiny_model.add_hook(
        name=activation_name_2,
        hook=partial(
            hook_fn_generic_patch_or_freeze,
            activation_name=activation_name_2,
            position=position,
            orig_cache=orig_cache,
            new_cache=new_cache,
        ),
    )
    _, cache = tiny_model.run_with_cache(orig_tokens, return_type=None)
    assert t.equal(cache[activation_name_1], orig_cache[activation_name_1])
    assert t.equal(cache[activation_name_2], new_cache[activation_name_2])


def test_hook_fn_add_activation_to_ctx(tiny_dataset_and_model):
    (
        tiny_model,
        orig_tokens,
        _,
        _,
        _,
        _,
        _,
    ) = tiny_dataset_and_model
    activation_name = "blocks.4.hook_resid_post"
    tiny_model.add_hook(
        name=activation_name,
        hook=partial(hook_fn_add_activation_to_ctx, user_def_name="saved_acts"),
    )
    _, cache = tiny_model.run_with_cache(orig_tokens, return_type=None)
    assert t.equal(
        cache[activation_name], tiny_model.hook_dict[activation_name].ctx["saved_acts"]
    )


def test_single_path_patch_logits(tiny_dataset_and_model):
    (
        tiny_model,
        orig_tokens,
        new_tokens,
        orig_logits,
        _,
        new_logits,
        _,
    ) = tiny_dataset_and_model
    sender = ("blocks.1.hook_resid_post",)
    receiver = ("blocks.4.hook_resid_pre",)
    seq_pos = -1
    patched_logits, _ = single_path_patch(
        codebook_model=tiny_model,
        orig_input=orig_tokens,
        new_input=new_tokens,
        sender_name=sender,
        receiver_name=receiver,
        seq_pos=seq_pos,
        test_mode=True,
    )

    assert not t.equal(patched_logits, orig_logits)
    assert not t.equal(patched_logits, new_logits)


def test_single_path_patch_cache(tiny_dataset_and_model):
    (
        tiny_model,
        orig_tokens,
        new_tokens,
        _,
        orig_cache,
        _,
        new_cache,
    ) = tiny_dataset_and_model
    sender = ("blocks.1.hook_resid_post",)
    receiver = ("blocks.4.hook_resid_pre",)
    seq_pos = -1
    _, cache = single_path_patch(
        codebook_model=tiny_model,
        orig_input=orig_tokens,
        new_input=new_tokens,
        sender_name=sender,
        receiver_name=receiver,
        seq_pos=seq_pos,
        test_mode=True,
    )

    assert t.equal(cache[sender], orig_cache[sender])
    assert not t.equal(cache[receiver], new_cache[receiver])
    assert not t.equal(cache[receiver], orig_cache[receiver])

    for hook_name, cache in cache.items():
        if "blocks" in hook_name:
            hook_layer = int(hook_name.split(".")[1])
            receiver_layer = int(receiver[0].split(".")[1])
            if hook_layer < receiver_layer:
                assert t.equal(cache, orig_cache[hook_name])
            if hook_layer > receiver_layer:
                if seq_pos is None:
                    assert not t.equal(cache, orig_cache[hook_name])
                assert not t.equal(cache, new_cache[hook_name])
