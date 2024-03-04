from copy import deepcopy
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util import nethook
from util.generate import generate_fast

from .compute_u import compute_u
from .compute_v import compute_v
from .rome_hparams import ROMEHyperParams

CONTEXT_TEMPLATES_CACHE = None


def apply_rome_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict], # List of requests to apply to the model
    hparams: ROMEHyperParams,
    copy=False,
    return_orig_weights=False,
) -> Tuple[AutoModelForCausalLM, List[str]]:
    """
    Returns a model with the desired changes.

    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.

    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    if copy:
        model = deepcopy(model)#creates a deep copy of the model. This is done to ensure that the original model is not modified during the editing process. The deep copy is stored in the model variable.

    weights_copy = {} #creates an empty dictionary to store the original weights that changed. This dictionary will be used to store the original weights of the model that were changed during the editing process.

    for i, request in enumerate(requests): #iterates through the list of requests to apply to the model. The index and the request are stored in the variables i and request respectively.
        deltas = execute_rome(model, tok, request, hparams) ## Execute the ROME method on the model with the given request and hyperparameters

        with torch.no_grad():
            for w_name, (delta_u, delta_v) in deltas.items():    #iterates through the dictionary of deltas. The weight name and the delta values are stored in the variables w_name and (delta_u, delta_v) respectively.
                upd_matrix = delta_u.unsqueeze(1) @ delta_v.unsqueeze(0)     #computes the update matrix by taking the outer product of delta_u and delta_v. outer product results in a matrix where each element is the product of the corresponding elements from the two input tensors
                w = nethook.get_parameter(model, w_name)    #retrieves the weight tensor from the model using the weight name. The weight tensor is stored in the variable w.
                upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)    #ensures that the update matrix has the same shape as the weight tensor. If the update matrix does not have the same shape as the weight tensor, a ValueError is raised.

                if return_orig_weights and w_name not in weights_copy:  #checks if the return_orig_weights flag is set to True and if the weight name is not in the weights_copy dictionary.
                    assert i == 0
                    weights_copy[w_name] = w.detach().clone()   #stores a deep copy of the weight tensor in the weights_copy dictionary. The weight tensor is stored using the weight name as the key.

                w[...] += upd_matrix    #updates the weight tensor by adding the update matrix to it. The update matrix is added to the weight tensor in place.

        print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy


def execute_rome(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: ROMEHyperParams,
) -> Dict[str, Tuple[torch.Tensor]]: #It returns a dictionary (deltas) containing tensors representing left and right vectors for each updated weight parameter.
    """
    Executes the ROME update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    # Update target and print info
    request = deepcopy(request)
    if request["target_new"]["str"][0] != " ":
        # Space required for correct tokenization
        request["target_new"]["str"] = " " + request["target_new"]["str"]
    print(
        f"Executing ROME algorithm for the update: "
        f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
    )

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()} #v.detach().clone():

    #For each weight tensor v in the original weights dictionary, the operations detach() and clone() are applied.
    #detach(): This method creates a new tensor that shares the same data as the original tensor but is detached from the computation graph. It prevents any further gradient computation on v.
    #clone(): This method creates a copy of the detached tensor. The new tensor is a separate copy with its own memory.

    # Update loop: sequentially intervene at each specified layer
    deltas = {}
    for layer in sorted(hparams.layers):
        # Compute rank-1 update matrix
        left_vector: torch.Tensor = compute_u(
            model,
            tok,
            request,
            hparams,
            layer,
            get_context_templates(model, tok, hparams.context_template_length_params),
        )
        print("Left vector shape:", left_vector.shape)
        right_vector: torch.Tensor = compute_v(
            model,
            tok,
            request,
            hparams,
            layer,
            left_vector,
            get_context_templates(model, tok, hparams.context_template_length_params),
        )
        print("Right vector shape:", right_vector.shape)

        with torch.no_grad():
            # Determine correct transposition of delta matrix
            weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            upd_matrix = left_vector.unsqueeze(1) @ right_vector.unsqueeze(0)
            upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

            # Update model weights and record desired changes in `delta` variable
            weights[weight_name][...] += upd_matrix
            deltas[weight_name] = (
                left_vector.detach(),
                right_vector.detach(),
            )

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by ROME does not match original weight shape. "
            "Check for bugs in the code?"
        )


def get_context_templates(model, tok, length_params):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = ["{}"] + [
            x + ". {}"
            for x in sum(   #The sum function is used to concatenate all the generated text snippets into a single list.
                (
                    generate_fast(
                        model,
                        tok,
                        ["<|endoftext|>"],
                        n_gen_per_prompt=n_gen,
                        max_out_len=length,
                    )
                    for length, n_gen in length_params
                ),
                [],
            )
        ]

        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE
