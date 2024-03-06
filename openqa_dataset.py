from abc import ABC
import json
from collections import OrderedDict
import random
import torch
from torch.utils.data import Dataset, DataLoader


def print_rank_0(message):
    """If distributed is initialized print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


class DistributedBatchSampler(data.sampler.BatchSampler):
    """Similar to normal implementation of distributed sampler, except
    implementation is at the batch sampler level, instead of just the
    sampler level. This allows wrapping of arbitrary data samplers
    (sequential, random, WeightedRandomSampler, etc.) with this batch
    sampler.

    The `interleave` argument specifies how to distribute a batch. A value
    of True combined with the above random sampler is equivalent to pytorch's
    torch.utils.data.distributed.DistributedSampler.

    For the following batch [0,1,2,3,4,5,6,7] and data parallelism of 2
    specifying True will result in the following samples for each gpu:
        GPU0: [0,2,4,6] GPU1: [1,3,5,7]
    specifying False will result in the following samples:
        GPU0: [0,1,2,3] GPU1: [4,5,6,7]"""

    def __init__(self, sampler, batch_size, drop_last, rank=-1,
                 world_size=2, wrap_last=False, interleave=False):
        super(DistributedBatchSampler, self).__init__(sampler, batch_size,
                                                      drop_last)
        if rank == -1:
            assert False, 'should not be here'
            rank = torch.distributed.get_rank()
        self.rank = rank
        self.world_size = world_size
        self.sampler.wrap_around = 0
        self.wrap_around = 0
        self.wrap_last = wrap_last
        self.start_iter = 0
        self.interleave = interleave

    def __iter__(self):
        batch = []
        i = 0
        for idx in self.data_iterator(self.sampler, wrap_around=False):
            batch.append(idx)
            if len(batch) == self.batch_size:
                tbatch = self._batch(batch)
                if i >= self.start_iter:
                    yield tbatch
                    self.start_iter = 0
                i += 1
                batch = []
        batch_len = len(batch)
        if batch_len > 0 and not self.drop_last:
            if self.wrap_last:
                self.sampler.wrap_around -= (self.batch_size)
                self.wrap_around += (len(batch))
                self.wrap_around %= self.batch_size
            yield self._batch(batch)
        if self.wrap_last:
            self.sampler.wrap_around += self.batch_size

    def data_iterator(self, _iter, wrap_around=False):
        """iterates through data and handles wrap around"""
        for i, idx in enumerate(_iter):
            if i < self.wrap_around % self.batch_size:
                continue
            if wrap_around:
                self.wrap_around += 1
                self.wrap_around %= self.batch_size
            yield idx

    def _batch(self, batch):
        """extracts samples only pertaining to this worker's batch"""
        if self.interleave:
            return batch[self.rank:self.batch_size:self.world_size]
        start = self.rank * self.batch_size // self.world_size
        end = (self.rank + 1) * self.batch_size // self.world_size
        return batch[start:end]


def get_openqa_dataset(task_name, dataset_path, sample_rate=1.0):
    dataset = OpenQADataset(task_name,
                            "open-domain retrieval",
                            dataset_path,
                            sample_rate)
    return dataset


class OpenQADataset(ABC, Dataset):
    def __init__(self, task_name, dataset_name, filepath, sample_rate):
        self.task_name = task_name
        self.dataset_name = dataset_name
        print_rank_0(' > building {} dataset for {}:'.format(self.task_name,
                                                             self.dataset_name))
        self.samples = self.load_dataset(filepath)

        if sample_rate < 1:  # subsample
            k = int(len(self.samples) * sample_rate)
            self.samples = random.sample(self.samples, k)

        print_rank_0('  >> total number of samples: {}'.format(len(self.samples)))
        n_contexts = len(self.samples[0]["ctxs"]) if "ctxs" in self.samples[0] else len(self.samples[0]["contexts"])
        print_rank_0('  >> total number of contexts in a sample: {}'.format(n_contexts))

        if "trivia" in filepath or 'webq' in filepath or 'entity-questions' in filepath \
                or "BEIR" in filepath or "squad" in filepath or "mia" in filepath or "xor" in filepath \
                or "XOR" in filepath:
            self.ques_punc = ""
        elif "nq" in filepath or "efficientqa" in filepath:
            self.ques_punc = "?"
        else:
            # raise ValueError("invalid dataset name {}".format(filepath))
            print_rank_0("dataset name {} not found. do nothing".format(filepath))
            self.ques_punc = ""

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]

        # These [CLS] and [SEP] tokens exist due to BERT tokenization, so we need to remove them
        if "[CLS]" and "[SEP]" in row['question']:
            row['question'] = " ".join(row['question'].split()[1:-1])

        if self.task_name == "reranking":
            decoder_prompt = "Question: {}{}".format(row['question'], self.ques_punc)
        else:
            raise ValueError("invalid --task-name argument {}".format(self.task_name))

        encoder_contexts = None
        if 'ctxs' in row:
            encoder_contexts = row['ctxs']
        elif 'contexts' in row:
            encoder_contexts = row['contexts']

        answers = row['answers']

        sample = {'id': row.get('q_id', row.get("id", idx)),
                  'encoder_ids': encoder_contexts,
                  'decoder_ids': decoder_prompt,
                  'question': row['question'],
                  'answers': answers,
                  'lang': row.get('lang', 'en').strip()}
        return sample

    @staticmethod
    def load_dataset(filepath):
        with open(filepath) as fp:
            data = json.load(fp)

        # condition for interfacing with pyserineni BM25 outputs
        if isinstance(data, dict):
            return list(data.values())
        else:
            return data


def get_one_epoch_dataloader(dataset, args, batch_size=None):
    """Specifically one epoch to be used in an indexing job."""
    # args = get_args()

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    if batch_size is None:
        batch_size = args.batch_size

    global_batch_size = batch_size * world_size
    num_workers = args.num_workers

    sampler = torch.utils.data.SequentialSampler(dataset)

    # importantly, drop_last must be False to get all the data.
    batch_sampler = DistributedBatchSampler(sampler,
                                            batch_size=global_batch_size,
                                            drop_last=False,
                                            rank=rank,
                                            world_size=world_size)

    # Data loader. Note that batch size is the per GPU batch size.
    data_loader = CustomDataLoader(dataset,
                                   batch_sampler=batch_sampler,
                                   num_workers=num_workers,
                                   pin_memory=True)
    return data_loader


class CustomDataLoader(DataLoader):
    def __init__(self, dataset, eval=False, **kwargs):
        if kwargs.get('collate_fn', None) is None:
            kwargs['collate_fn'] = self._collate_fn
        self.eval = eval
        super().__init__(dataset, **kwargs)

    def _collate_fn(self, batch_data):
        # generate batch
        batch_size = len(batch_data)
        if batch_size == 0:
            raise StopIteration
        tensorized = OrderedDict()
        for d in batch_data:
            for k, v in d.items():
                tensorized.setdefault(k, []).append(v)
        assert len(tensorized) == 5 or len(tensorized) == 6

        return tensorized
