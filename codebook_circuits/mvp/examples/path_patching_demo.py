import torch as t
from codebook_circuits.mvp.codebook_path_patching import single_path_patch
from codebook_circuits.mvp.data import TravelToCityDataset
from codebook_circuits.mvp.utils import (
    compute_average_logit_difference,
    setup_pythia410M_hooked_model,
    time_a_fn,
)

# Set up model
cb_model = setup_pythia410M_hooked_model()

## Creating some data for testing and development
dataset = TravelToCityDataset(n_examples=50)
responses = dataset.correct_incorrects
incorrect_correct_toks = t.cat(
    [cb_model.to_tokens(response, prepend_bos=False).T for response in responses]
)
orig_input = dataset.clean_prompts
new_input = dataset.corrupted_prompts

# Benchmarking Original Model Performance
orig_logits = cb_model(orig_input)
original_perfomance = compute_average_logit_difference(
    orig_logits, incorrect_correct_toks, answer_position=-1
)

print(f"Original Model Perfomance: {original_perfomance}")

_, orig_cache = cb_model.run_with_cache(orig_input, return_type=None)
_, new_cache = cb_model.run_with_cache(new_input, return_type=None)

# Path Patch over given sender-receiver path
# Timing Results
# test_mode = False: 5.33 seconds
# test mode = True: 10.88 seconds

path_patched_logits, _ = single_path_patch(
    codebook_model=cb_model,
    orig_input=orig_input,
    new_input=new_input,
    sender_name=("cb_5", 17),
    receiver_name=("cb_10", 19),
    orig_cache=orig_cache,
    new_cache=new_cache,
    seq_pos=None,
    test_mode=True,
)

path_patched_perf = compute_average_logit_difference(
    path_patched_logits, incorrect_correct_toks, answer_position=-1
)

print(f"Patched Model Perfomance - Proper Method: {path_patched_perf}")
