#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
BiEncoder component + loss function for 'all-in-batch' training
"""

import logging
import random
from typing import Tuple, List
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor as T
from torch import nn

from dpr.utils.data_utils import Tensorizer
from dpr.utils.data_utils import normalize_question

logger = logging.getLogger(__name__)


@dataclass
class BiEncoderBatch:
    question_ids: T
    question_segments: T
    context_ids: T
    ctx_segments: T
    is_positive: List
    hard_negatives: List

    def _asdict(self):
        return self.__dict__


@dataclass
class BiEncoderBatchWithScore:
    question_ids: T
    question_segments: T
    context_ids: T
    ctx_segments: T
    context_scores: T

    def _asdict(self):
        return self.__dict__


def dot_product_scores(q_vectors: T, ctx_vectors: T) -> T:
    """
    calculates q->ctx scores for every row in ctx_vector
    :param q_vector:
    :param ctx_vector:
    :return:
    """
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    r = torch.matmul(q_vectors, torch.transpose(ctx_vectors, 0, 1))
    return r


def dot_product_scores_per_question(q_vectors: T, ctx_vectors: T) -> T:
    """
    calculates q->ctx scores for each question in the batch
    contexts are not shared across questions
    :param q_vector: shape (batch_size, vector_dim)
    :param ctx_vector: shape (batch_size, num_contexts, vector_dim)
    :return: logits: Tensor of shape (batch_size, num_contexts)
    """
    # q_vector: n1 x D, ctx_vectors: n1 x nc x D, result n1 x nc
    r = torch.bmm(q_vectors.unsqueeze(1), ctx_vectors.transpose(1, 2)).squeeze(1)
    return r


def cosine_scores(q_vector: T, ctx_vectors: T):
    # q_vector: n1 x D, ctx_vectors: n2 x D, result n1 x n2
    return F.cosine_similarity(q_vector, ctx_vectors, dim=1)


class BiEncoder(nn.Module):
    """ Bi-Encoder model component. Encapsulates query/question and context/passage encoders.
    """

    def __init__(self, question_model: nn.Module, ctx_model: nn.Module, fix_q_encoder: bool = False,
                 fix_ctx_encoder: bool = False):
        super(BiEncoder, self).__init__()
        self.question_model = question_model
        self.ctx_model = ctx_model
        self.fix_q_encoder = fix_q_encoder
        self.fix_ctx_encoder = fix_ctx_encoder

    @staticmethod
    def get_representation(
        sub_model: nn.Module,
        ids: T,
        segments: T,
        attn_mask: T,
        fix_encoder: bool = False
    ) -> Tuple[T, T, T]:
        sequence_output = None
        pooled_output = None
        hidden_states = None
        if ids is not None:
            if fix_encoder:
                with torch.no_grad():
                    sequence_output, pooled_output, hidden_states = sub_model(ids, segments, attn_mask)

                if sub_model.training:
                    sequence_output.requires_grad_(requires_grad=True)
                    pooled_output.requires_grad_(requires_grad=True)
            else:
                sequence_output, pooled_output, hidden_states = sub_model(ids, segments, attn_mask)

        return sequence_output, pooled_output, hidden_states

    def forward(self, question_ids: T, question_segments: T, question_attn_mask: T, context_ids: T, ctx_segments: T,
                ctx_attn_mask: T) -> Tuple[T, T]:

        _q_seq, q_pooled_out, _q_hidden = self.get_representation(self.question_model, question_ids, question_segments,
                                                                  question_attn_mask, self.fix_q_encoder)
        _ctx_seq, ctx_pooled_out, _ctx_hidden = self.get_representation(self.ctx_model, context_ids, ctx_segments,
                                                                        ctx_attn_mask, self.fix_ctx_encoder)

        return q_pooled_out, ctx_pooled_out
    
    @classmethod
    def get_input_create_fn(
            cls,
            tensorizer: Tensorizer,
            insert_title: bool,
            num_hard_negatives: int = 0,
            num_other_negatives: int = 0,
            shuffle: bool = True,
            shuffle_positives: bool = False,
    ):
        def fn(samples: List):
            return cls.create_biencoder_input(
                samples, tensorizer, insert_title,
                num_hard_negatives=num_hard_negatives,
                num_other_negatives=num_other_negatives,
                shuffle=shuffle,
                shuffle_positives=shuffle_positives,
            )
        return fn

    @classmethod
    def create_biencoder_input(cls,
                               samples: List,
                               tensorizer: Tensorizer,
                               insert_title: bool,
                               num_hard_negatives: int = 0,
                               num_other_negatives: int = 0,
                               shuffle: bool = True,
                               shuffle_positives: bool = False,
                               ) -> BiEncoderBatch:
        """
        Creates a batch of the biencoder training tuple.
        :param samples: list of data items (from json) to create the batch for
        :param tensorizer: components to create model input tensors from a text sequence
        :param insert_title: enables title insertion at the beginning of the context sequences
        :param num_hard_negatives: amount of hard negatives per question (taken from samples' pools)
        :param num_other_negatives: amount of other negatives per question (taken from samples' pools)
        :param shuffle: shuffles negative passages pools
        :param shuffle_positives: shuffles positive passages pools
        :return: BiEncoderBatch tuple
        """
        question_tensors = []
        ctx_tensors = []
        positive_ctx_indices = []
        hard_neg_ctx_indices = []

        for sample in samples:
            # ctx+ & [ctx-] composition
            # as of now, take the first(gold) ctx+ only
            if shuffle and shuffle_positives:
                positive_ctxs = sample['positive_ctxs']
                positive_ctx = positive_ctxs[np.random.choice(len(positive_ctxs))]
            else:
                positive_ctx = sample['positive_ctxs'][0]

            neg_ctxs = sample['negative_ctxs']
            hard_neg_ctxs = sample['hard_negative_ctxs']
            question = normalize_question(sample['question'])

            if shuffle:
                random.shuffle(neg_ctxs)
                random.shuffle(hard_neg_ctxs)

            neg_ctxs = neg_ctxs[0:num_other_negatives]
            hard_neg_ctxs = hard_neg_ctxs[0:num_hard_negatives]

            all_ctxs = [positive_ctx] + neg_ctxs + hard_neg_ctxs
            hard_negatives_start_idx = 1
            hard_negatives_end_idx = 1 + len(hard_neg_ctxs)

            current_ctxs_len = len(ctx_tensors)

            texts = [
                ctx['text'][0]
                if isinstance(ctx['text'], list) and len(ctx['text']) == 1
                else ctx['text']
                for ctx in all_ctxs
            ]

            if all(['title' not in ctx for ctx in all_ctxs]):
                titles = None
            else:
                titles = [ctx.get('title', '') for ctx in all_ctxs]
            
            sample_ctxs_tensors = tensorizer.texts_to_tensors(texts=texts, titles=titles if insert_title else None)

            ctx_tensors.extend(list(sample_ctxs_tensors))
            positive_ctx_indices.append(current_ctxs_len)
            hard_neg_ctx_indices.append(
                [
                    i for i in
                    range(current_ctxs_len + hard_negatives_start_idx, current_ctxs_len + hard_negatives_end_idx)
                ]
            )

            question_tensors.append(tensorizer.text_to_tensor(question))

        ctxs_tensor = torch.cat([ctx.view(1, -1) for ctx in ctx_tensors], dim=0)
        questions_tensor = torch.cat([q.view(1, -1) for q in question_tensors], dim=0)

        ctx_segments = torch.zeros_like(ctxs_tensor)
        question_segments = torch.zeros_like(questions_tensor)

        return BiEncoderBatch(
            questions_tensor, question_segments, ctxs_tensor, ctx_segments,
            positive_ctx_indices, hard_neg_ctx_indices
        )

    @classmethod
    def create_biencoder_input_with_score(
        cls,
        samples: List,
        tensorizer: Tensorizer,
        insert_title: bool,
        num_contexts: int = 16,
        shuffle: bool = False,
    ) -> BiEncoderBatchWithScore:
        """
        Creates a batch of the biencoder training tuple.
        :param samples: list of data items (from json) to create the batch for
        :param tensorizer: components to create model input tensors from a text sequence
        :param insert_title: enables title insertion at the beginning of the context sequences
        :param num_context: number of contexts per question
        :param shuffle: whether to randomly pick contexts (up to num_context) from the sample
        :return: BiEncoderBatch tuple
        """
        question_tensors = []
        ctx_tensors = []
        context_scores = []

        for sample in samples:
            question = normalize_question(sample['question'])
            all_contexts = sample['ctxs']
            if shuffle:
                random.shuffle(all_contexts)

            all_contexts = all_contexts[:num_contexts]
            all_scores = torch.tensor([float(ctx["score"]) for ctx in all_contexts])

            texts = [
                ctx['text'][0]
                if isinstance(ctx['text'], list) and len(ctx['text']) == 1
                else ctx['text']
                for ctx in all_contexts
            ]
            titles = [ctx['title'] for ctx in all_contexts]
            sample_ctxs_tensors = tensorizer.texts_to_tensors(texts=texts, titles=titles if insert_title else None)

            ctx_tensors.append(sample_ctxs_tensors)  # (num_contexts, seq_length)
            question_tensors.append(tensorizer.text_to_tensor(question))  # (seq_length)
            context_scores.append(all_scores)  # (num_contexts)

        ctxs_tensor = torch.stack(ctx_tensors, dim=0)  # (batch_size, num_contexts, seq_length)
        questions_tensor = torch.stack(question_tensors, dim=0)  # (batch_size, seq_length)
        scores_tensor = torch.stack(context_scores, dim=0)  # (batch_size, num_contexts)

        ctx_segments = torch.zeros_like(ctxs_tensor)
        question_segments = torch.zeros_like(questions_tensor)

        return BiEncoderBatchWithScore(questions_tensor, question_segments, ctxs_tensor, ctx_segments, scores_tensor)


class BiEncoderNllLoss(object):
    def calc(self, q_vectors: T, ctx_vectors: T, positive_idx_per_question: list,
             hard_negatice_idx_per_question: list = None) -> Tuple[T, int]:
        """
        Computes nll loss for the given lists of question and ctx vectors.
        Note that although hard_negatice_idx_per_question in not currently in use, one can use it for the
        loss modifications. For example - weighted NLL with different factors for hard vs regular negatives.
        :return: a tuple of loss value and amount of correct predictions per batch
        """
        scores = self.get_scores(q_vectors, ctx_vectors)

        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)

        softmax_scores = F.log_softmax(scores, dim=1)

        loss = F.nll_loss(softmax_scores, torch.tensor(positive_idx_per_question).to(softmax_scores.device),
                          reduction='mean')

        max_score, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = (max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)).sum()
        return loss, correct_predictions_count

    @staticmethod
    def get_scores(q_vector: T, ctx_vectors: T) -> T:
        f = BiEncoderNllLoss.get_similarity_function()
        return f(q_vector, ctx_vectors)

    @staticmethod
    def get_similarity_function():
        return dot_product_scores


class BiEncoderKLLoss(object):
    def calc(self, q_vectors: T, ctx_vectors: T, ctx_scores: T, temperature: float = 1.0) -> T:
        """
        Computes kl loss for the given lists of question and ctx vectors.
        :param q_vectors: shape (batch_size, vector_dim)
        :param ctx_vectors: shape (batch_size, n_contexts, vector_dim)
        :param ctx_scores: shape (batch_size, n_contexts)
        :param temperature: temperature for the softmax
        :return: a tuple of loss value and amount of correct predictions per batch
        """
        scores = self.get_scores(q_vectors, ctx_vectors)

        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)

        ctx_scores = ctx_scores * temperature
        softmax_scores = F.log_softmax(scores, dim=1)
        softmax_ctx_scores = F.log_softmax(ctx_scores, dim=1)

        loss = F.kl_div(
            softmax_scores,
            softmax_ctx_scores.to(softmax_scores.device),
            reduction='batchmean',
            log_target=True
        )
        return loss

    @staticmethod
    def get_scores(q_vector: T, ctx_vectors: T) -> T:
        f = BiEncoderKLLoss.get_similarity_function()
        return f(q_vector, ctx_vectors)

    @staticmethod
    def get_similarity_function():
        return dot_product_scores_per_question


class BiEncoderInBatchKLLoss(object):
    def calc(self, q_vectors: T, ctx_vectors: T, ctx_scores: T, temperature: float = 1.0) -> T:
        """
        Computes kl loss for the given lists of question and ctx vectors.
        Contexts from other questions are used as in-batch negatives
        :param q_vectors: shape (batch_size, vector_dim)
        :param ctx_vectors: shape (batch_size, n_contexts, vector_dim)
        :param ctx_scores: shape (batch_size, n_contexts)
        :param temperature: temperature for softmax
        :return: loss value
        """
        batch_size, n_contexts, vector_dim = ctx_vectors.size()
        ctx_vectors = ctx_vectors.view(-1, vector_dim)
        scores = self.get_scores(q_vectors, ctx_vectors)  # (batch_size, batch_size * n_contexts)

        ctx_scores = ctx_scores * temperature
        softmax_ctx_scores = F.log_softmax(ctx_scores, dim=1)
        ctx_scores_inbatch = torch.ones_like(scores) * -1000.0
        for i in range(batch_size):
            ctx_scores_inbatch[i, i * n_contexts:(i + 1) * n_contexts] = softmax_ctx_scores[i]

        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)

        softmax_scores = F.log_softmax(scores, dim=1)

        loss = F.kl_div(
            softmax_scores,
            ctx_scores_inbatch,
            reduction='batchmean',
            log_target=True
        )
        return loss

    @staticmethod
    def get_scores(q_vector: T, ctx_vectors: T) -> T:
        f = BiEncoderInBatchKLLoss.get_similarity_function()
        return f(q_vector, ctx_vectors)

    @staticmethod
    def get_similarity_function():
        return dot_product_scores
