import torch
import torch.nn as nn


def resize_token_embeddings(old_tok_embeddings, num_new_tokens):
    old_num_tokens, embedding_dim = old_tok_embeddings.weight.shape
    new_num_tokens = old_num_tokens + num_new_tokens

    # Create a new embedding layer
    new_tok_embeddings = nn.Embedding(
        new_num_tokens,
        embedding_dim,
        device=old_tok_embeddings.weight.device,
        dtype=old_tok_embeddings.weight.dtype)

    with torch.no_grad():
        new_tok_embeddings.weight[:old_num_tokens, :] = old_tok_embeddings.weight

    return new_tok_embeddings

def resize_output_projection(old_output_proj, num_new_tokens):
    # old_output_proj: nn.Linear of shape [embed_dim, old_vocab_size]
    old_num_tokens, embedding_dim = old_output_proj.weight.shape
    new_num_tokens = old_num_tokens + num_new_tokens

    new_output_proj = nn.Linear(
        embedding_dim,
        new_num_tokens,
        device=old_output_proj.weight.device,
        dtype=old_output_proj.weight.dtype,
        bias=False)

    with torch.no_grad():
        new_output_proj.weight[:old_num_tokens, :] = old_output_proj.weight

    return new_output_proj