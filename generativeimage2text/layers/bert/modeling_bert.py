# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model. """

from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import math
from io import open

import torch
from torch import nn

from .activations import ACT2FN
from .modeling_utils import (PretrainedConfig, prune_linear_layer)

logger = logging.getLogger()

BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json",
}

class QK2Attention(nn.Module):
    def forward(self, query, key, attention_mask, gamma):
        return qk2attn(query, key, attention_mask, gamma)

def qk2attn(query, key, attention_mask, gamma):
    query = query / gamma
    attention_scores = torch.matmul(query, key.transpose(-1, -2))
    if attention_mask is not None:
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask
    return attention_scores.softmax(dim=-1)


class BertConfig(PretrainedConfig):
    pretrained_config_archive_map = BERT_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 vocab_size_or_config_json_file=30522,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 **kwargs):
        super(BertConfig, self).__init__(**kwargs)
        if isinstance(vocab_size_or_config_json_file, str):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

LayerNormClass = torch.nn.LayerNorm

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.softmax = nn.Softmax(dim=-1)
        self.qk2attn = QK2Attention()

    def transpose_for_scores(self, x):
        if torch._C._get_tracing_state():
            # exporter is not smart enough to detect dynamic size for some paths
            x = x.view(x.shape[0], -1, self.num_attention_heads, self.attention_head_size)
        else:
            new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
            x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, head_mask=None,
            history_state=None):
        if history_state is not None:
            x_states = torch.cat([history_state, hidden_states], dim=1)
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(x_states)
            mixed_value_layer = self.value(x_states)
        else:
            mixed_query_layer = self.query(hidden_states)
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_probs = self.qk2attn(query_layer, key_layer, attention_mask, math.sqrt(self.attention_head_size))
        #attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        #attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        #attention_scores = attention_scores + attention_mask
        #attention_probs = self.softmax(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.pre_norm = hasattr(config, 'pre_norm') and config.pre_norm
        if not self.pre_norm:
            self.LayerNorm = LayerNormClass(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if not self.pre_norm:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        else:
            hidden_states = hidden_states + input_tensor
        return hidden_states

class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.pre_norm = hasattr(config, 'pre_norm') and config.pre_norm
        if self.pre_norm:
            self.LayerNorm = LayerNormClass(config.hidden_size, eps=config.layer_norm_eps)
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        for head in heads:
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)
        # Update hyper params
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads

    def forward(self, input_tensor, attention_mask, head_mask=None,
            history_state=None):
        if self.pre_norm:
            self_outputs = self.self(self.LayerNorm(input_tensor), attention_mask, head_mask,
                    self.layerNorm(history_state) if history_state else history_state)
        else:
            self_outputs = self.self(input_tensor, attention_mask, head_mask,
                    history_state)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.pre_norm = hasattr(config, 'pre_norm') and config.pre_norm
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if not self.pre_norm:
            self.LayerNorm = LayerNormClass(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        if not self.pre_norm:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        else:
            hidden_states = hidden_states + input_tensor
        return hidden_states

class Mlp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pre_norm = hasattr(config, 'pre_norm') and config.pre_norm
        self.intermediate = BertIntermediate(config)
        if self.pre_norm:
            self.LayerNorm = LayerNormClass(config.hidden_size, eps=config.layer_norm_eps)
        self.output = BertOutput(config)

    def forward(self, attention_output):
        if not self.pre_norm:
            intermediate_output = self.intermediate(attention_output)
        else:
            intermediate_output = self.intermediate(self.LayerNorm(attention_output))
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.pre_norm = hasattr(config, 'pre_norm') and config.pre_norm
        self.use_mlp_wrapper = hasattr(config, 'use_mlp_wrapper') and config.use_mlp_wrapper
        self.attention = BertAttention(config)
        if self.use_mlp_wrapper:
            self.mlp = Mlp(config)
        else:
            self.intermediate = BertIntermediate(config)
            if self.pre_norm:
                self.LayerNorm = LayerNormClass(config.hidden_size, eps=config.layer_norm_eps)
            self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, head_mask=None,
                history_state=None):
        attention_outputs = self.attention(hidden_states, attention_mask,
                head_mask, history_state)
        attention_output = attention_outputs[0]
        if self.use_mlp_wrapper:
            layer_output = self.mlp(attention_output)
        else:
            if not self.pre_norm:
                intermediate_output = self.intermediate(attention_output)
            else:
                intermediate_output = self.intermediate(self.LayerNorm(attention_output))
            layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + attention_outputs[1:]  # add attentions if we output them
        return outputs

class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.pre_norm = hasattr(config, 'pre_norm') and config.pre_norm
        if self.pre_norm:
            self.LayerNorm = LayerNormClass(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask, head_mask=None,
                encoder_history_states=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            history_state = None if encoder_history_states is None else encoder_history_states[i]
            layer_outputs = layer_module(
                hidden_states, attention_mask,
                (None if head_mask is None else head_mask[i]),
                history_state,
            )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
        if self.pre_norm:
            hidden_states = self.LayerNorm(hidden_states)
        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # outputs, (hidden states), (attentions)

