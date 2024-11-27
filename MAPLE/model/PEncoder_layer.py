from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.modules.gelu import gelu, gelu_accurate
from model.modules.layer_norm import LayerNorm
from model.modules.multihead_attention import MultiheadAttention
from model.modules.quant_noise import quant_noise
from model.modules.fairseq_dropout import FairseqDropout
from torch import Tensor

def get_activation_fn(activation: str) -> Callable:
    """ Returns the activation function corresponding to `activation` """
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return gelu
    elif activation == "gelu_accurate":
        return gelu_accurate
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    else:
        raise RuntimeError("--activation-fn {} not supported".format(activation))

class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """
    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False, no_encoder_att=False, layernum=0
    ):
        super().__init__()
        self.embed_dim = getattr(args, 'encoder_embed_dim', 256)
        self.dropout_module = nn.Dropout(p=0.5)
        self.no_encoder_attn=no_encoder_attn
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)
        self.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
        self.no_encoder_att = no_encoder_att

        self.cross_self_attention = getattr(args, "cross_self_attention", False)
        # The argument of insertCausalSelfAttn should only be used with the NAT model training
        # Otherwise the unmasked self attention will leak info
        self.insertCausalSelfAttn = getattr(args, 'insertCausalSelfAttn', True)
        self.noLN = getattr(args, 'dont_use_layernorm', False)

        if self.insertCausalSelfAttn:
            self.self_attn_unmasked = self.build_self_attention(
                self.embed_dim,
                args,
                add_bias_kv=add_bias_kv,
                add_zero_attn=add_zero_attn,
            )
        else:
            self.self_attn_unmasked = None

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )


        self.activation_fn = get_activation_fn(
            activation=str(args.activation_fn) if getattr(args, "activation_fn", None) is not None else "relu"
        )
        self.normalize_before = args.encoder_normalize_before  # usually equal False

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)

        self.fc1 = self.build_fc1(
            self.embed_dim, self.encoder_ffn_embed_dim, self.quant_noise, self.quant_noise_block_size
        )
        self.fc2 = self.build_fc2(
            self.encoder_ffn_embed_dim, self.embed_dim, self.quant_noise, self.quant_noise_block_size
        )

        if not self.noLN:
            self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)
            self.final_layer_norm = LayerNorm(self.embed_dim, export=export)

        if self.insertCausalSelfAttn and not self.noLN:
            self.self_attn_unmasked_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.need_attn = True
        print(self.no_encoder_attn, self.noLN)

        if self.no_encoder_attn and not self.noLN:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.onnx_trace = False

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(self, embed_dim, args, add_bias_kv=False, add_zero_attn=False):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,  # 0.1
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not getattr(args, "cross_self_attention", False),
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def build_encoder_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, 'decoder_embed_dim', 256),
            vdim=getattr(args, 'decoder_embed_dim', 256),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def forward(self, x, input,
        self_attn_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """

        if need_head_weights:
            need_attn = True

        # if self.self_attn_unmasked is not None:
        #
        #     residual = x
        #     if not self.noLN:
        #         x = self.self_attn_unmasked_layer_norm(x)
        #     x, attn = self.self_attn_unmasked(
        #         query=x,
        #         key=x,
        #         value=x,
        #         need_weights=need_attn,
        #         need_head_weights=need_head_weights
        #     )
        #     x = self.dropout_module(x)
        #     x = residual + x
        #     if not self.normalize_before and not self.noLN:
        #         x = self.self_attn_unmasked_layer_norm(x)

        if self.self_attn is not None:
            residual = x
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                need_weights=False,
                attn_mask=self_attn_mask
            )

            x = self.dropout_module(x)
            x = residual + x
            if not self.normalize_before and not self.noLN:
                x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None:
            residual = x
            if self.normalize_before and not self.noLN:
                x = self.encoder_attn_layer_norm(x)
            x, attn = self.encoder_attn(
                query=x,
                key=input,
                value=input,
                need_weights=need_attn,
                need_head_weights=need_head_weights,
            )
            x = self.dropout_module(x)
            x = residual + x
            if not self.normalize_before and not self.noLN:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before and not self.noLN:
            x = self.final_layer_norm(x)

        x = self.activation_fn(self.fc1(x))
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before and not self.noLN:
            x = self.final_layer_norm(x)
        return x, attn, None

def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m