"""
Load pre-trained Llama model weights following torchtune.
"""
import gc
from typing import Any, Dict

import torch
from torchtune.models import convert_weights
from torchtune.training.checkpointing._utils import (
    safe_torch_load,
    ModelType,
)
from torchtune import training
from torchtune.utils import get_logger, get_world_size_and_rank, log_rank_zero
import torchtune

MODEL_CONFIG_DICT = {
    "meta-llama/Llama-3.2-1B-Instruct": {
        "num_attention_heads": 32,
        "num_hidden_layers": 16,
        "num_key_value_heads": 8,
        "hidden_size": 2048,
        "head_dim": 64,
    },
    "Qwen/Qwen2.5-14B-Instruct": {
        "architectures": [
            "Qwen2ForCausalLM"
        ],
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "eos_token_id": 151645,
        "hidden_act": "silu",
        "hidden_size": 5120,
        "initializer_range": 0.02,
        "intermediate_size": 13824,
        "max_position_embeddings": 32768,
        "max_window_layers": 70,
        "model_type": "qwen2",
        "num_attention_heads": 40,
        "num_hidden_layers": 48,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-06,
        "rope_theta": 1000000.0,
        "sliding_window": 131072,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.43.1",
        "use_cache": True,
        "use_sliding_window": False,
        "vocab_size": 152064
        }
}

# def load_checkpoint() -> Dict[str, Any]:
#     """
#     Load HF checkpoint from file.

#     The keys and weights from across all checkpoint files are merged into a single state_dict.
#     We preserve the "state_dict key" <-> "checkpoint file" mapping in weight_map so we can
#     write the state dict correctly in ``save_checkpoint``.

#     Before returning, the model state dict is converted to a torchtune-compatible format using
#     the appropriate convert_weights function (depending on ``self._model_type``).

#     Returns:
#         state_dict (Dict[str, Any]): torchtune checkpoint state dict

#     Raises:
#         ValueError: If the values in the input state_dict are not Tensors
#     """

#     _weight_map = {}

#     # merged state_dict contains keys and weights from all the checkpoint files
#     merged_state_dict: Dict[str, torch.Tensor] = {}

#     # converted_state_dict is the final state_dict passed to the recipe after the
#     # keys are converted into the torchtune format. This optionally also contains
#     # the recipe state and adapter weights
#     converted_state_dict: Dict[str, Dict[str, torch.Tensor]] = {}

#     # _checkpoint_paths are already sorted so simply enumerate to generate the right id
#     for cpt_idx, cpt_path in enumerate(checkpoint_paths):
#         state_dict = safe_torch_load(cpt_path)
#         for key, value in state_dict.items():
#             # Ensure that the state dict is a flat dict of keys and tensors. Breaking this assumption
#             # will break recipe code
#             if not isinstance(value, torch.Tensor):
#                 raise ValueError(
#                     f"Expected all values in the state dict to be torch.Tensor. "
#                     f"Found {type(value)} instead."
#                 )
#             # idx is written in the 4 digit format (eg: 0001, 0002, etc.)
#             _weight_map[key] = f"{cpt_idx + 1:04}"
#         merged_state_dict.update(state_dict)

#         # delete the state_dict to free up memory; TODO check if this del is needed
#         del state_dict
#         gc.collect()
#     # if self._model_type in (ModelType.PHI3_MINI, ModelType.PHI4):
#     #     log_rank_zero(
#     #         logger=logger,
#     #         msg="Converting Phi weights from HF format."
#     #         "Note that conversion of adapter weights into PEFT format is not supported.",
#     #     )
#     #     from torchtune.models.phi3._convert_weights import phi3_hf_to_tune

#     #     num_heads = self._config["num_attention_heads"]
#     #     num_kv_heads = self._config["num_key_value_heads"]
#     #     dim = self._config["hidden_size"]

#     #     # Should only pass num_heads, num_kv_heads, dim for GQA
#     #     if num_heads == num_kv_heads:
#     #         num_heads, num_kv_heads, dim = None, None, None

#     #     converted_state_dict[training.MODEL_KEY] = phi3_hf_to_tune(
#     #         merged_state_dict,
#     #         num_heads=num_heads,
#     #         num_kv_heads=num_kv_heads,
#     #         dim=dim,
#     #     )

    

#     cfg_dict = MODEL_CONFIG_DICT[model_name]
#     converted_state_dict[training.MODEL_KEY] = qwen2_hf_to_tune(
#         merged_state_dict,
#         num_heads=cfg_dict["num_attention_heads"],
#         num_kv_heads=cfg_dict["num_key_value_heads"],
#         dim=cfg_dict["hidden_size"],
#         tie_word_embeddings=cfg_dict["tie_word_embeddings"],
#     )


#     return converted_state_dict

def load_checkpoint(
    ckpt_path: str,
    model_name: str,
) -> Dict[str, Any]:
    """
    Load HF checkpoint from file.

    The keys and weights from across all checkpoint files are merged into a single state_dict.
    We preserve the "state_dict key" <-> "checkpoint file" mapping in weight_map so we can
    write the state dict correctly in ``save_checkpoint``.

    Before returning, the model state dict is converted to a torchtune-compatible format using
    the appropriate convert_weights function (depending on ``self._model_type``).

    Args:
        ckpt_path: the path that store the model checkpoint downloaded from huggingface `original`
        model_name: the HF model name, such as `meta-llama/Llama-3.2-1B-Instruct`
    Returns:
        state_dict (Dict[str, Any]): torchtune checkpoint state dict

    Raises:
        ValueError: If the values in the input state_dict are not Tensors
    """

    # merged state_dict contains keys and weights from all the checkpoint files
    merged_state_dict: Dict[str, torch.Tensor] = {}

    # converted_state_dict is the final state_dict passed to the recipe after the
    # keys are converted into the torchtune format. This optionally also contains
    # the recipe state and adapter weights
    converted_state_dict: Dict[str, Dict[str, torch.Tensor]] = {}

    # _checkpoint_paths are already sorted so simply enumerate to generate the right id
    # state_dict = safe_torch_load(ckpt_path)

    for cpt_idx, cpt_path in enumerate(ckpt_path):
        state_dict = safe_torch_load(cpt_path)
        for key, value in state_dict.items():
            # Ensure that the state dict is a flat dict of keys and tensors. Breaking this assumption
            # will break recipe code
            if not isinstance(value, torch.Tensor):
                raise ValueError(
                    f"Expected all values in the state dict to be torch.Tensor. "
                    f"Found {type(value)} instead."
                )
        merged_state_dict.update(state_dict)
    # merged_state_dict.update(state_dict)

    # delete the state_dict to free up memory; TODO check if this del is needed
    del state_dict
    gc.collect()

    cfg_dict = MODEL_CONFIG_DICT[model_name]

    from torchtune.models.qwen2._convert_weights import qwen2_hf_to_tune
    converted_state_dict = qwen2_hf_to_tune(
        merged_state_dict,
        num_heads=cfg_dict["num_attention_heads"],
        num_kv_heads=cfg_dict["num_key_value_heads"],
        dim=cfg_dict["hidden_size"],
        tie_word_embeddings=cfg_dict["tie_word_embeddings"],
    )

    return converted_state_dict



