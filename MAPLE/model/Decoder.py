# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
import torch.nn.functional as F
from model.Decoder_layer import TransformerDecoderLayer
from model.modules.layer_norm import LayerNorm
from model.modules.layer_drop import LayerDropModuleList



def fill_with_neg_inf(t):
    """FP16-compatible function that fills a tensor with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


def _argmax(x, dim):
    return (x == x.max(dim, keepdim=True)[0]).type_as(x)


def new_arange(x, *size):
    """
    Return a Tensor of `size` filled with a range function on the device of x.
    If size is empty, using the size of the variable x.
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()


def _uniform_assignment(src_lens, trg_lens):
    max_trg_len = trg_lens.max()
    steps = (src_lens.float() - 1) / (trg_lens.float() - 1)  # step-size
    index_t = new_arange(trg_lens, max_trg_len).float()
    index_t = steps[:, None] * index_t[None, :]  # batch_size X max_trg_len
    index_t = torch.round(index_t).long().detach()
    return index_t


# def build_decoder(cls, args, tgt_dict, embed_tokens):
#     decoder = NATransformerDecoder(args, tgt_dict, embed_tokens)
#     return decoder
#
#     # decoding
#     word_ins_out = self.decoder(normalize=False, prev_output_tokens=prev_output_tokens,
#         encoder_out=encoder_out)



class TransformerDecoder(nn.Module):
    def __init__(self, args, embed_dim, no_encoder_attn=False):
        super(TransformerDecoder, self).__init__()
        self.embed_dim = embed_dim
        self.decoder_layerdrop = getattr(args, 'decoder_layerdrop', 0.0)
        self.insertCausalSelfAttn = getattr(args, 'insertCausalSelfAttn', True)
        self._future_mask = torch.empty(0)
        self.dropout_module = nn.Dropout(p=0.5)

        if self.decoder_layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.decoder_layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_decoder_layer(args, i)
                for i in range(args.decoder_layer_num)
            ]
        )

        self.num_layers = len(self.layers)

        if args.decoder_normalize_before and not getattr(
            args, "no_decoder_final_norm", False
        ):
            self.layer_norm = LayerNorm(self.embed_dim)
        else:
            self.layer_norm = None

    def build_decoder_layer(self, args, layernum=0):
        return TransformerDecoderLayer(args, layernum=layernum)

    def forward(self, encoder_out, prev_output_tokens):
        encoder_out = encoder_out.detach()
        features, inner_state = self.extract_features(prev_output_tokens, encoder_out=encoder_out)
        return features

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out=None,
    ):


        # B x T x C -> T x B x C
        x = prev_output_tokens.transpose(0, 1)
        attn = None
        inner_states = [x]

        if self.insertCausalSelfAttn:
            for i, layer in enumerate(self.layers):
                # for layers in insertCausalSelfAttn, every layer needs self attention mask
                self_attn_mask = self.buffered_future_mask(x)
                x, attn, _ = layer(
                    x,
                    encoder_out,
                    self_attn_mask=self_attn_mask
                )
                inner_states.append(x)
        else:
            for i, layer in enumerate(self.layers):
                x, attn, _ = layer(
                    x,
                    encoder_out,
                    self_attn_mask=None
                )
                inner_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)
        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, inner_states

    # causal self attention
    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        # self._future_mask.device != tensor.device is not working in TorchScript. This is a workaround.
        if (
            self._future_mask.size(0) == 0
            or (not self._future_mask.device == tensor.device)
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(
                fill_with_neg_inf(torch.zeros([dim, dim])), 1
            )
        self._future_mask = self._future_mask.to(tensor)
        return self._future_mask[:dim, :dim]



    def forward_copying_source(self, src_embeds, src_masks, tgt_masks):
        length_sources = src_masks.sum(1)
        length_targets = tgt_masks.sum(1)
        mapped_inputs = _uniform_assignment(length_sources, length_targets).masked_fill(
            ~tgt_masks, 0
        )
        copied_embedding = torch.gather(
            src_embeds,
            1,
            mapped_inputs.unsqueeze(-1).expand(
                *mapped_inputs.size(), src_embeds.size(-1)
            ),
        )
        return copied_embedding
