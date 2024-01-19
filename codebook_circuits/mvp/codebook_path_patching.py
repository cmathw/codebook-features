from functools import partial
from typing import List, Optional, Tuple, Union

import torch as t
from codebook_circuits.mvp.more_tl_mods import get_act_name
from codebook_circuits.mvp.utils import time_a_fn, logit_change_metric
from codebook_features.models import HookedTransformerCodebookModel
from jaxtyping import Float, Int
from torch import Tensor
from transformer_lens import ActivationCache
from transformer_lens.hook_points import HookPoint
from torch import zeros
from tqdm import tqdm

# This implementation takes inspiration from here:
# https://github.com/callummcdougall/path_patching/blob/main/path_patching.py


def hook_fn_generic_patching_from_context(
    activation: Float[Tensor, "batch pos ..."],
    hook: HookPoint,
    user_def_name: str = "activation",
) -> Float[Tensor, "batch pos ..."]:
    """
    Patches activations from hook context, if they are there.

    Args:
        activation (Float[Tensor, "batch pos ..."]): Activation to patch
        hook (HookPoint): Hook to patch from
        name (str): Name of the activation in the hook's context

    Return:
        Float[Tensor, "batch pos ..."]: Updated Activation
    """
    # Check if the user defined name is in the hook context
    if user_def_name in hook.ctx:
        # If it is, patch the activation with the corresponding activation in hook context
        activation[:] = hook.ctx[user_def_name][:]
    else:
        raise ValueError(f"Name {user_def_name} not found in hook context")
    return activation


def hook_fn_generic_patch_or_freeze(
    activation: Float[Tensor, "batch pos ..."],
    activation_name: str,
    position: int,
    hook: HookPoint,
    orig_cache: ActivationCache,
    new_cache: ActivationCache,
) -> Float[Tensor, "batch pos ..."]:
    """
    Patch over the codebook at a specific layer and head index with a new codebook (ie. a codebook associated with a corrupted prompt run)

    Args:
        activation (Float[Tensor, "batch pos ..."]): Activation to patch over
        activation_name (str): Name of the activation to patch over
        position (int): Sequence position to patch over
        hook (HookPoint): TransformerLens Hook Point
        new_cache (ActivationCache): New Activation Cache (associated with a corrupted prompt run for example)

    Return:
        Float[Tensor, "batch pos ..."]: New Activation
    """
    # Copy over the original codebook to avoid overwriting
    activation[...] = orig_cache[hook.name][...]
    if hook.name == activation_name:
        # This should only change the cache at the specified position, leaving the orig_cache copy elsewhere
        activation[:, position] = new_cache[hook.name][:, position]
    return activation


def hook_fn_add_activation_to_ctx(
    activation: Float[Tensor, "batch pos ..."], hook: HookPoint, user_def_name: str
) -> Float[Tensor, "batch pos codes"]:
    """
    Enter an activation with a pre-defined name into the hook's context.

    Args:
        activation (Float[Tensor, "batch pos ..."]): Activation to add to the hook's context
        hook (HookPoint): Hook to add the activation to
        user_def_name (str): Name to give the activation in the hook's context

    Return:
        Float[Tensor, "batch pos codes"]: Activation added to hook's context
    """
    hook.ctx[user_def_name] = activation
    return activation


def get_orig_new_cache_and_logits(
    codebook_model: HookedTransformerCodebookModel,
    orig_input: Union[str, List[str], Int[Tensor, "batch pos"]],
    new_input: Union[str, List[str], Int[Tensor, "batch pos"]],
) -> Tuple[ActivationCache, ActivationCache]:
    """
    Get original and new cache and logits from a model run with cache

    Args:
        codebook_model (HookedTransformer): Model to run with cache
        orig_input (Union[str, List[str], Int[Tensor, "batch pos"]]): Original input to the model
        new_input (Union[str, List[str], Int[Tensor, "batch pos"]]): New input to the model

    Returns:
        Tuple[ActivationCache, ActivationCache, Float[Tensor, "batch pos d_vocab"], Float[Tensor, "batch pos d_vocab"]]: Original and new cache and logits
    """
    # Run model with cache to get original and new cache and logits
    orig_logits, orig_cache = codebook_model.run_with_cache(
        orig_input, return_type="logits"
    )
    new_logits, new_cache = codebook_model.run_with_cache(
        new_input, return_type="logits"
    )

    # Make sure logits not associated with computational graph to save memory
    orig_logits = orig_logits.detach()
    new_logits = new_logits.detach()

    return orig_cache, orig_logits, new_cache, new_logits


def get_activation_name(
    name: Tuple[str, Optional[Union[int, str]], Optional[str]]
) -> str:
    """
    Get the full activation name from a tuple of activation name components
    Args:
        name (Tuple[str, Optional[Union[int, str]], Optional[str]]): Tuple of activation name components

    Returns:
        str: Full activation name
    """

    return get_act_name(*name) if len(name) > 1 else get_act_name(name[0])


# time function
# @time_a_fn
def single_path_patch(
    codebook_model: HookedTransformerCodebookModel,
    orig_input: Union[str, List[str], Int[Tensor, "batch pos"]],
    new_input: Union[str, List[str], Int[Tensor, "batch pos"]],
    sender_name: Tuple[str, Optional[Union[int, str]], Optional[str]],
    receiver_name: Tuple[str, Optional[Union[int, str]], Optional[str]],
    seq_pos: Optional[Int[Tensor, "batch pos"]] = None,
    orig_cache: Optional[ActivationCache] = None,
    new_cache: Optional[ActivationCache] = None,
    test_mode: bool = False,
) -> Union[Float[Tensor, "batch pos d_vocab"], ActivationCache]:
    """
    Implement an easy test version of path patching between two codebooks.
    This implementation only patches one path (from a single codebook to another single codebook).
    It patches across either all sequence positions for each codebook or a single sequence position for each codebook.

    Args:
        codebook_model: Model on which we are performing path patching.
        orig_input: Original input to the model (ie. clean inputs)
        new_input: New input to the model (ie. corrupt inputs)
        sender_name: The node for which we are patching from given as either:
            - Full activation name (same as key found in Activation Cache)
            eg. (blocks.0.attn.codebook_layer.codebook.1.hook_codebook_ids)
            - Abbreviated activation name
            eg. ("cb_1", 2) for codebook head idx 1, layer 2
        receiver_name:  The node for which we are patching to given as either
            - Full activation name (same as key found in Activation Cache)
            eg. (blocks.23.hook_resid_post)
            - Abbreviated activation name
            eg. ("resid_post", 23).
        seq_pos: Sequence position to patch over. If None, patch over all sequence positions.
        orig_cache: Original cache to use for patching. Should be supplied if not running test_mode.
        new_cache: New cache to use for patching. Should be supplied if not running test_mode.
        test_mode:
            If True, will run in test mode and return patched logits and cache.
            If False, will run in production mode and return patched logits only.
            It is slow because we calculate both orig and new activations for each codebook (instead of providing)
            these caches from the outset. We also include many assert comparisons to verify that activations are what we expect them to be.
            NOTE: Test mode takes approximately 2x as long to run.

    Returns:
        Patched Logits: The logits associated with the model run on the original input but
        new input patched over the single path defined between sender_codebooks -> receiver_codebooks.
    """

    # Clear hooks and context for extra safety
    codebook_model.reset_hooks()

    # Get original and new cache objects
    if test_mode:
        orig_cache, orig_logits, new_cache, new_logits = get_orig_new_cache_and_logits(
            codebook_model, orig_input, new_input
        )

    if not test_mode and (orig_cache is None or new_cache is None):
        raise ValueError("If not in test mode, must provide orig_cache and new_cache")

    # Get codebook activation names
    sender_activation_name = get_activation_name(sender_name)
    receiver_activation_name = get_activation_name(receiver_name)

    # Store the original and new activations for the sender and receiver codebooks.
    # We do this for testing purposes.
    if test_mode:
        sender_orig_activations = orig_cache[sender_activation_name]
        receiver_orig_activations = orig_cache[receiver_activation_name]
        sender_new_activations = new_cache[sender_activation_name]
        receiver_new_activations = new_cache[receiver_activation_name]

    # Add hook to patch sender activations and freeze rest
    codebook_model.add_hook(
        name=sender_activation_name,
        hook=partial(
            hook_fn_generic_patch_or_freeze,
            activation_name=sender_activation_name,
            position=seq_pos,
            orig_cache=orig_cache,
            new_cache=new_cache,
        ),
    )

    # Add hook to add sender activations to context
    if test_mode:
        codebook_model.add_hook(
            name=sender_activation_name,
            hook=partial(
                hook_fn_add_activation_to_ctx, user_def_name="sender_activations_pre"
            ),
        )

    # Add hook to add receiver activations to context
    codebook_model.add_hook(
        name=receiver_activation_name,
        hook=partial(
            hook_fn_add_activation_to_ctx, user_def_name="receiver_activations_pre"
        ),
    )

    # Run with hooks to populate contexts with appropriate activations (those associate with sender patch)
    _ = codebook_model.run_with_hooks(orig_input, return_type=None)

    # Get activations from context for comparison asserts
    if test_mode:
        sender_activations_pre = codebook_model.hook_dict[sender_activation_name].ctx[
            "sender_activations_pre"
        ]
        receiver_activations_pre = codebook_model.hook_dict[
            receiver_activation_name
        ].ctx["receiver_activations_pre"]

        # The sender activations should be equal to the new activations, the receiver activations should NOT be equal to the original activations OR equal to the new activations
        assert t.equal(
            sender_activations_pre[:, seq_pos], sender_new_activations[:, seq_pos]
        )
        assert not t.equal(receiver_activations_pre, receiver_orig_activations)
        assert not t.equal(receiver_activations_pre, receiver_new_activations)

    # Clear out the previous hooks but keep the context
    codebook_model.reset_hooks(clear_contexts=False)

    # Patch in the receiver with activations defined earlier as receiver_cache
    codebook_model.add_hook(
        name=receiver_activation_name,
        hook=partial(
            hook_fn_generic_patching_from_context,
            user_def_name="receiver_activations_pre",
        ),
    )

    # Add hook to add sender activations to context
    if test_mode:
        codebook_model.add_hook(
            name=sender_activation_name,
            hook=partial(
                hook_fn_add_activation_to_ctx, user_def_name="sender_activations_post"
            ),
        )

        # Add hook to add receiver activations to context
        codebook_model.add_hook(
            name=receiver_activation_name,
            hook=partial(
                hook_fn_add_activation_to_ctx, user_def_name="receiver_activations_post"
            ),
        )

    # Get patched logits and cache
    patched_logits, patched_cache = codebook_model.run_with_cache(
        orig_input, return_type="logits"
    )

    # Make sure logits not associated with computational graph to save memory
    patched_logits = patched_logits.detach()

    # Get activations from context for comparison asserts
    if test_mode:
        sender_activations_post = codebook_model.hook_dict[sender_activation_name].ctx[
            "sender_activations_post"
        ]
        receiver_activations_post = codebook_model.hook_dict[
            receiver_activation_name
        ].ctx["receiver_activations_post"]

        # The sender activations should be back to the original activations, the receiver activations should equal neither new or original activations
        assert t.equal(sender_activations_post, sender_orig_activations)
        assert not t.equal(receiver_activations_post, receiver_orig_activations)
        assert not t.equal(receiver_activations_post, receiver_new_activations)

        # We expect all activations up to the receiver to be equal to the original activations
        # We expect all activations after the receiver to NOT be equal to either original or new activations
        for hook_name, cache in patched_cache.items():
            if "blocks" in hook_name:
                hook_layer = int(hook_name.split(".")[1])
                receiver_layer = int(receiver_activation_name.split(".")[1])
                if hook_layer < receiver_layer:
                    assert t.equal(cache, orig_cache[hook_name])
                if hook_layer > receiver_layer:
                    if seq_pos is None:
                        assert not t.equal(cache, orig_cache[hook_name])
                    assert not t.equal(cache, new_cache[hook_name])

        # We expect that patched logits will not be equal to either the original or new logits
        assert not t.equal(patched_logits, orig_logits)
        assert not t.equal(patched_logits, new_logits)

    # Clear hooks  and contextfor safety
    codebook_model.reset_hooks(clear_contexts=True)
    if test_mode:
        return patched_logits, patched_cache

    return patched_logits


def iter_codebook_patch_patching(
    codebook_model: HookedTransformerCodebookModel,
    orig_input: Float[Tensor, "batch pos d_model"],
    new_input: Union[str, List[str], Int[Tensor, "batch pos"]],
    incorrect_correct_toks: Int[Tensor, "batch 2"],
    response_position: int,
    patch_position: int,
    receiver_name: Tuple[str, Optional[Union[int, str]], Optional[str]] = ("resid_post", 23),
) -> Float[Tensor, "layer cb_head_idx"]:
    """
    Iteratively activation patch over each codebook for a given position in each layer and head index.

    Args:
        orig_tokens (Tensor): Original tokens (ie. clean tokens), shape (batch, pos, d_model)
        new_cache (ActivationCache): New Activation Cache (associated with a corrupted prompt run for example)
        position (int): Position to Patch Over

    Returns:
        Float[Tensor, "layer cb_head_idx"]: A performance metric for each layer and head index
    """
    orig_logits, orig_cache = codebook_model.run_with_cache(orig_input)
    _, new_cache = codebook_model.run_with_cache(new_input)
    n_layers = codebook_model.cfg.n_layers
    n_heads = codebook_model.cfg.n_heads
    codebook_patch_array = zeros(
        n_layers,
        n_heads,
        dtype=t.float32,
        device=codebook_model.cfg.device,
        requires_grad=False,
    )
    for sender_layer in tqdm(range(n_layers), desc="Iterating over Model Layers"):
        for sender_head_idx in range(n_heads):
            patched_logits = single_path_patch(
                codebook_model=codebook_model,
                orig_input=orig_input,
                new_input=new_input,
                sender_name=(f"cb_{sender_head_idx}", sender_layer),
                receiver_name=receiver_name,
                seq_pos=patch_position,
                orig_cache=orig_cache,
                new_cache=new_cache,
                test_mode=False
            )

            perf_metric = logit_change_metric(
                orig_logits=orig_logits,
                new_logits=patched_logits,
                incorrect_correct_toks=incorrect_correct_toks,
                answer_position=response_position,
            )

            codebook_patch_array[sender_layer, sender_head_idx] = perf_metric

    return codebook_patch_array