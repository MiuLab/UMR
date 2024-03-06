#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Command line tool that produces embeddings for a large documents base based on the pretrained ctx & question encoders
 Supposed to be used in a 'sharded' way to speed up the process.
"""
import os
import pathlib

import argparse
import csv
import logging
import pickle
from typing import List, Tuple

import numpy as np
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import transformers
from torch import nn
from torch.cuda.amp import autocast
from tqdm import tqdm

from dpr.models import init_biencoder_components
from dpr.options import add_encoder_params, setup_args_gpu, print_args, set_encoder_params_from_state, \
    add_tokenizer_params, add_cuda_params
from dpr.utils.data_utils import Tensorizer
from dpr.utils.model_utils import setup_for_distributed_mode, get_model_obj, load_states_from_checkpoint, move_to_device


logger = logging.getLogger()
logger.setLevel(logging.INFO)
if (logger.hasHandlers()):
    logger.handlers.clear()
console = logging.StreamHandler()
logger.addHandler(console)
transformers.logging.set_verbosity_error()


class CtxDataset(torch.utils.data.Dataset):
    def __init__(self, ctx_rows: List[Tuple[object, str, str]], tensorizer: Tensorizer, insert_title: bool = True):
        self.rows = ctx_rows
        self.tensorizer = tensorizer
        self.insert_title = insert_title

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, item):
        ctx = self.rows[item]
        # r = self.tensorizer.text_to_tensor(ctx[1], title=ctx[2] if self.insert_title else None)
        return ctx

    def collate(self, ctxs):
        texts = [ctx[1] for ctx in ctxs]
        titles = [ctx[2] for ctx in ctxs]
        return self.tensorizer.texts_to_tensors(texts, titles=titles if self.insert_title else None)


def no_op_collate(xx: List[object]):
    return xx


def gen_ctx_vectors(
    output_file: str,
    ctx_rows: List[Tuple[object, str, str]],
    model: nn.Module,
    tensorizer: Tensorizer,
    insert_title: bool = True,
    fp16: bool = False
) -> List[Tuple[object, np.array]]:
    bsz = args.batch_size
    total = 0

    dataset = CtxDataset(ctx_rows, tensorizer, insert_title)
    loader = torch.utils.data.DataLoader(
        dataset, shuffle=False, num_workers=4, collate_fn=dataset.collate,
        drop_last=False, batch_size=bsz, pin_memory=True
    )

    with open(output_file, mode='wb') as f:
        for batch_id, batch_token_tensors in enumerate(tqdm(loader)):
            ctx_ids_batch = move_to_device(batch_token_tensors, args.device)
            ctx_seg_batch = move_to_device(torch.zeros_like(ctx_ids_batch), args.device)
            ctx_attn_mask = move_to_device(tensorizer.get_attn_mask(ctx_ids_batch), args.device)

            with torch.no_grad():
                if fp16:
                    with autocast():
                        _, out, _ = model(ctx_ids_batch, ctx_seg_batch, ctx_attn_mask)
                else:
                    _, out, _ = model(ctx_ids_batch, ctx_seg_batch, ctx_attn_mask)

            out = out.float().cpu()
            batch_start = batch_id * bsz
            ctx_ids = [r[0] for r in ctx_rows[batch_start:batch_start + bsz]]

            assert len(ctx_ids) == out.size(0)

            total += len(ctx_ids)
            results = [
                (ctx_ids[i], out[i].view(-1).numpy())
                for i in range(out.size(0))
            ]
            pickle.dump(results, f)

    return total


def main(args):
    if args.model_file and os.path.exists(args.model_file):
        saved_state = load_states_from_checkpoint(args.model_file)
        set_encoder_params_from_state(saved_state.encoder_params, args)
    elif not args.model_file:
        logger.warning("args.model_file is not set, assuming we are loading a pretrained model from Huggingface Hub")
        saved_state = None
    else:
        raise ValueError(
            "args.model_file is not found in local file system. If you want to load a pretrained model from \
            Huggingface Hub, set args.pretrained_model_cfg instead"
        )

    print_args(args)

    tensorizer, encoder, _ = init_biencoder_components(args.encoder_model_type, args, inference_only=True)

    encoder = encoder.ctx_model

    encoder, _ = setup_for_distributed_mode(encoder, None, args.device, args.n_gpu, args.local_rank)
    encoder.eval()

    if saved_state:
        # load weights from the model file
        model_to_load = get_model_obj(encoder)
        logger.info('Loading saved model state ...')
        logger.debug('saved model keys =%s', saved_state.model_dict.keys())

        prefix_len = len('ctx_model.')
        ctx_state = {
            key[prefix_len:]: value
            for (key, value) in saved_state.model_dict.items()
            if key.startswith('ctx_model.')
        }
        model_to_load.load_state_dict(ctx_state)

    logger.info('reading data from file=%s', args.ctx_file)

    rows = []
    with open(args.ctx_file) as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        # file format: doc_id, doc_text, title
        rows.extend([(row[0], row[1], row[2]) for row in reader if row[0] != 'id'])

    shard_size = int(len(rows) / args.num_shards)
    start_idx = args.shard_id * shard_size
    end_idx = start_idx + shard_size

    logger.info('Producing encodings for passages range: %d to %d (out of total %d)', start_idx, end_idx, len(rows))
    rows = rows[start_idx:end_idx]

    filename = args.out_file + '_' + str(args.shard_id) + '.pkl'
    pathlib.Path(os.path.dirname(filename)).mkdir(parents=True, exist_ok=True)
    logger.info('Writing results to %s' % filename)
    total = gen_ctx_vectors(filename, rows, encoder, tensorizer, insert_title=True, fp16=args.fp16)

    logger.info('Total passages processed %d. Written to %s', total, filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    add_encoder_params(parser)
    add_tokenizer_params(parser)
    add_cuda_params(parser)

    parser.add_argument('--ctx_file', type=str, default=None, help='Path to passages set .tsv file')
    parser.add_argument('--out_file', required=True, type=str, default=None,
                        help='output file path to write results to ')
    parser.add_argument('--shard_id', type=int, default=0, help="Number(0-based) of data shard to process")
    parser.add_argument('--num_shards', type=int, default=1, help="Total amount of data shards")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for the passage encoder forward pass")
    args = parser.parse_args()

    setup_args_gpu(args)

    main(args)
