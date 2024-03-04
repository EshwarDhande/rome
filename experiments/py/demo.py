import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from baselines.ft import FTHyperParams, apply_ft_to_model
from rome import ROMEHyperParams, apply_rome_to_model
from util import nethook
from util.generate import generate_fast
from util.globals import *


def demo_model_editing(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    generation_prompts: List[str],
    alg_name: str = "ROME",
) -> Tuple[AutoModelForCausalLM, Dict[str, torch.Tensor]]:
    """
    Applies the selected model editing algorithm. Generates text both before and after
    for comparison of model behavior.
    Returns the updated model and the original values of
    weights that were changed.
    """

    nethook.set_requires_grad(True, model) # Ensure that we can edit the model. likely done to enable gradient computation for the model.

    RewritingParamsClass, apply_method, hparams_prefix, hparams_suffix = load_alg(
        alg_name
    ) #loads the algorithm-specific parameters and methods using the load_alg function. The returned values include the class for rewriting parameters (RewritingParamsClass), the method for applying the algorithm (apply_method), and prefixes/suffixes for hyperparameters.used to construct the path for loading hyperparameters. 
    # The RewritingParamsClass is a class that stores hyperparameters for the algorithm. The apply_method is a method that applies rewrites to the model. The hparams_prefix and hparams_suffix are used to locate the hyperparameters for the algorithm.
    # The load_alg function is used to load the dependencies for the desired algorithm. The dependencies include the class for storing
    # hyperparameters, the method for applying rewrites, the location of parameters, and a predefined suffix for the param file.
    #"rewriting parameters" could involve adjusting the weights and biases of a neural network, fine-tuning specific components, or modifying other model parameters
    
    
    params_name = (
        HPARAMS_DIR
        / hparams_prefix
        / f"{model.config._name_or_path.replace('/', '_')}{hparams_suffix}.json"
    ) #constructs the path for loading hyperparameters. The path is constructed using the hparams_prefix, model.config._name_or_path, and hparams_suffix. The hparams_prefix and hparams_suffix are used to locate the hyperparameters for the algorithm. The model.config._name_or_path is used to construct the path for loading hyperparameters.

    print_loud(f"Retrieving {alg_name} hyperparameters") #prints a string with # box for emphasis.
    print("Loading from", params_name) #prints the path for loading hyperparameters.
    hparams = RewritingParamsClass.from_json(params_name) #loads the hyperparameters for the algorithm. 
    #RewritingParamsClass: This is an instance of a class (or possibly a class itself) that represents the hyperparameters or configuration settings for a specific algorithm.
    #from_json: This is a method of the RewritingParamsClass. It suggests that the class has a method for loading its configuration from a JSON file.
    #params_name: This variable holds the path to the JSON file containing the hyperparameters. It is constructed based on the model's name, prefixes, and suffixes as part of the algorithm configuration.
    #So, this line is essentially loading the hyperparameters for the algorithm from a JSON file specified by params_name and storing them in the hparams variable.
    print(hparams) #prints the hyperparameters for the algorithm.

    print_loud("Generating pre-update text")
    pre_update_text = generate_fast(model, tok, generation_prompts, max_out_len=100)
    print(pre_update_text)

    print_loud(f"Applying {alg_name} to model")
    model_new, orig_weights = apply_method(
        model, tok, requests, hparams, return_orig_weights=True
    )

    print_loud("Generating post-update text")
    post_update_text = generate_fast(
        model_new, tok, generation_prompts, max_out_len=100
    )
    print(post_update_text)

    print_loud("Summarizing differences")
    for i, (prompt, pre, post) in enumerate(
        zip(generation_prompts, pre_update_text, post_update_text)
    ):#zip: This function is used to combine the elements of multiple iterables into a single iterable. It returns an iterator of tuples where the i-th tuple contains the i-th element from each of the argument sequences or iterables.
        if i > 0:
            print("".join(["-" for _ in range(10)]))  # separator

        prompt_str = "[Prompt]:" #prints the prompt string
        pre_str = f"[Pre-{alg_name}]:" #prints the pre-update string
        post_str = f"[Post-{alg_name}]:"
        pad_to = 1 + max(len(prompt_str), len(pre_str), len(post_str)) #pad_to: This variable holds the maximum length of the prompt, pre-update, and post-update strings. It is used to ensure that the strings are printed with the same length.

        for s, t in zip([prompt_str, post_str, pre_str], [prompt, post, pre]): #
            print(s.ljust(pad_to), t)  #prints each pair of string and text, left-justified with padding to the calculated pad_to value
    return model_new, orig_weights


def load_alg(alg_name):
    """
    Loads dependencies for the desired algorithm.
    Implementation is slightly awkward to prevent unnecessary imports on Colab.

    The return value is a tuple of the following:
    1. Class for storing hyperparameters
    2. Method for applying rewrites
    3. Location of parameters
    4. Predefined suffix for the param file
    """
    assert alg_name in [
        "FT",
        "FT-L",
        "FT-AttnEdit",
        "KN",
        "MEND",
        "MEND-CF",
        "MEND-zsRE",
        "KE",
        "KE-CF",
        "ROME",
    ]

    if alg_name == "ROME":
        return ROMEHyperParams, apply_rome_to_model, "ROME", ""
    elif "FT" in alg_name:
        d = {
            "FT": (FTHyperParams, apply_ft_to_model, "FT", "_unconstr"),
            "FT-AttnEdit": (FTHyperParams, apply_ft_to_model, "FT", "_attn"),
            "FT-L": (FTHyperParams, apply_ft_to_model, "FT", "_constr"),
        }
        return d[alg_name]
    else:
        from baselines.efk import EFKHyperParams, EfkRewriteExecutor
        from baselines.kn import KNHyperParams, apply_kn_to_model
        from baselines.mend import MENDHyperParams, MendRewriteExecutor

        d = {
            "KN": (KNHyperParams, apply_kn_to_model, "KN", ""),
            "MEND": (MENDHyperParams, MendRewriteExecutor().apply_to_model, "MEND", ""),
            "KE": (EFKHyperParams, EfkRewriteExecutor().apply_to_model, "KE", ""),
            "MEND-CF": (
                MENDHyperParams,
                MendRewriteExecutor().apply_to_model,
                "MEND",
                "_CF",
            ),
            "MEND-zsRE": (
                MENDHyperParams,
                MendRewriteExecutor().apply_to_model,
                "MEND",
                "_zsRE",
            ),
            "KE-CF": (
                EFKHyperParams,
                EfkRewriteExecutor().apply_to_model,
                "MEND",
                "_CF",
            ),
        }
        return d[alg_name]


def print_loud(x, pad=3):
    """
    Prints a string with # box for emphasis.

    Example:
    ############################
    #                          #
    #  Applying ROME to model  #
    #                          #
    ############################
    """

    n = len(x)
    print()
    print("".join(["#" for _ in range(n + 2 * pad)]))
    print("#" + "".join([" " for _ in range(n + 2 * (pad - 1))]) + "#")
    print(
        "#"
        + "".join([" " for _ in range(pad - 1)])
        + x
        + "".join([" " for _ in range(pad - 1)])
        + "#"
    )
    print("#" + "".join([" " for _ in range(n + 2 * (pad - 1))]) + "#")
    print("".join(["#" for _ in range(n + 2 * pad)]))


class StopExecution(Exception):
    def _render_traceback_(self):
        pass


def stop_execution():
    raise StopExecution
