#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Encoder model wrappers based on HuggingFace code
"""

import logging
from typing import Tuple, List

import torch
from torch import Tensor as T
from torch import nn
from transformers.models.bert import BertModel
from transformers import AdamW
from transformers import AutoTokenizer, AutoConfig

from dpr.utils.data_utils import Tensorizer
from .biencoder import BiEncoder
from .reader import Reader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)


def get_bert_biencoder_components(args, inference_only: bool = False, **kwargs):
    dropout = args.dropout if hasattr(args, 'dropout') else 0.0
    question_encoder = HFBertEncoder.init_encoder(args.pretrained_model_cfg,
                                                  projection_dim=args.projection_dim, dropout=dropout, **kwargs)
    ctx_encoder = HFBertEncoder.init_encoder(args.pretrained_model_cfg,
                                             projection_dim=args.projection_dim, dropout=dropout, **kwargs)

    fix_ctx_encoder = getattr(args, 'fix_ctx_encoder', False)
    print("fix context encoder: {}".format(fix_ctx_encoder))
    biencoder = BiEncoder(question_encoder, ctx_encoder, fix_ctx_encoder=fix_ctx_encoder)

    optimizer = get_optimizer(biencoder,
                              learning_rate=args.learning_rate,
                              adam_eps=args.adam_eps, weight_decay=args.weight_decay,
                              ) if not inference_only else None

    tensorizer = get_bert_tensorizer(args)

    return tensorizer, biencoder, optimizer


def get_bert_reader_components(args, inference_only: bool = False, **kwargs):
    dropout = args.dropout if hasattr(args, 'dropout') else 0.0
    encoder = HFBertEncoder.init_encoder(args.pretrained_model_cfg,
                                         projection_dim=args.projection_dim, dropout=dropout)

    hidden_size = encoder.config.hidden_size
    reader = Reader(encoder, hidden_size)

    optimizer = get_optimizer(reader,
                              learning_rate=args.learning_rate,
                              adam_eps=args.adam_eps, weight_decay=args.weight_decay,
                              ) if not inference_only else None

    tensorizer = get_bert_tensorizer(args)
    return tensorizer, reader, optimizer


def get_bert_tensorizer(args, tokenizer=None):
    if not tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_cfg, do_lower_case=args.do_lower_case, use_fast=False)
    return BertTensorizer(tokenizer, args.sequence_length)


def get_roberta_tensorizer(args, tokenizer=None):
    if not tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            args.pretrained_model_cfg, do_lower_case=args.do_lower_case, use_fast=False)
    return RobertaTensorizer(tokenizer, args.sequence_length)


def get_optimizer(model: nn.Module, learning_rate: float = 1e-5, adam_eps: float = 1e-8,
                  weight_decay: float = 0.0, ) -> torch.optim.Optimizer:
    no_decay = ['bias', 'LayerNorm.weight']

    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_eps)
    return optimizer


class HFBertEncoder(BertModel):
    def __init__(self, config, project_dim: int = 0):
        BertModel.__init__(self, config)
        assert config.hidden_size > 0, 'Encoder hidden_size can\'t be zero'
        self.encode_proj = nn.Linear(config.hidden_size, project_dim) if project_dim != 0 else None
        self.init_weights()

    @classmethod
    def init_encoder(cls, cfg_name: str, projection_dim: int = 0, dropout: float = 0.1, **kwargs) -> BertModel:
        cfg = AutoConfig.from_pretrained(cfg_name if cfg_name else 'bert-base-uncased')
        if dropout != 0:
            cfg.attention_probs_dropout_prob = dropout
            cfg.hidden_dropout_prob = dropout
        return cls.from_pretrained(cfg_name, config=cfg, project_dim=projection_dim, **kwargs)

    def forward(self, input_ids: T, token_type_ids: T, attention_mask: T) -> Tuple[T, ...]:
        outputs = super().forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )

        if self.config.architectures[0] == "Contriever":
            sequence_output = outputs.last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.)
            pooled_output = sequence_output.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        else:
            sequence_output = outputs.last_hidden_state
            pooled_output = sequence_output[:, 0, :]
        hidden_states = outputs.get("hidden_states", None)
        if self.encode_proj:
            pooled_output = self.encode_proj(pooled_output)
        return sequence_output, pooled_output, hidden_states

    def get_out_size(self):
        if self.encode_proj:
            return self.encode_proj.out_features
        return self.config.hidden_size


class BertTensorizer(Tensorizer):
    def __init__(self, tokenizer: AutoTokenizer, max_length: int, pad_to_max: bool = True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_to_max = pad_to_max

    def text_to_tensor(self, text: str, title: str = None, add_special_tokens: bool = True):
        if isinstance(text, list) and len(text) == 1:
            text = text[0]
        text = text.strip()

        # tokenizer automatic padding is explicitly disabled since its inconsistent behavior
        # FIXME: temporary enabling the tokenizer's truncation.
        if title:
            token_ids = self.tokenizer.encode(title, text_pair=text, add_special_tokens=add_special_tokens,
                                              max_length=self.max_length,
                                              pad_to_max_length=False, truncation=True)
        else:
            token_ids = self.tokenizer.encode(text, add_special_tokens=add_special_tokens, max_length=self.max_length,
                                              pad_to_max_length=False, truncation=True)

        seq_len = self.max_length
        if self.pad_to_max and len(token_ids) < seq_len:
            token_ids = token_ids + [self.tokenizer.pad_token_id] * (seq_len - len(token_ids))
        if len(token_ids) > seq_len:
            token_ids = token_ids[0:seq_len]
            token_ids[-1] = self.tokenizer.sep_token_id

        return torch.tensor(token_ids)

    def texts_to_tensors(self, texts: List[str], titles: List[str] = None, add_special_tokens: bool = True):
        texts = [text.strip() for text in texts]

        if titles:
            inputs = self.tokenizer(
                titles, text_pair=texts, add_special_tokens=add_special_tokens,
                max_length=self.max_length, padding=False, truncation=True
            )
        else:
            inputs = self.tokenizer(
                texts, add_special_tokens=add_special_tokens, max_length=self.max_length, padding=False, truncation=True
            )

        all_token_ids = inputs.input_ids

        for i in range(len(all_token_ids)):
            if len(all_token_ids[i]) > self.max_length:
                all_token_ids[i] = all_token_ids[i][0:self.max_length]
                all_token_ids[i][-1] = self.tokenizer.sep_token_id

        if self.pad_to_max:
            max_length = self.max_length
        else:
            max_length = max([len(token_ids) for token_ids in all_token_ids])

        all_token_ids = [
            token_ids + [self.tokenizer.pad_token_id] * (max_length - len(token_ids))
            for token_ids in all_token_ids
        ]

        return torch.tensor(all_token_ids)

    def get_pair_separator_ids(self) -> T:
        return torch.tensor([self.tokenizer.sep_token_id])

    def get_pad_id(self) -> int:
        return self.tokenizer.pad_token_id

    def get_attn_mask(self, tokens_tensor: T) -> T:
        return tokens_tensor != self.get_pad_id()

    def is_sub_word_id(self, token_id: int):
        token = self.tokenizer.convert_ids_to_tokens([token_id])[0]
        return token.startswith("##") or token.startswith(" ##")

    def to_string(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def set_pad_to_max(self, do_pad: bool):
        self.pad_to_max = do_pad


class RobertaTensorizer(BertTensorizer):
    def __init__(self, tokenizer, max_length: int, pad_to_max: bool = True):
        super(RobertaTensorizer, self).__init__(tokenizer, max_length, pad_to_max=pad_to_max)
