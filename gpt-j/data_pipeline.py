import gzip
import json
import os
import random
from typing import List, Tuple

import h5py
import numpy as np
import smdistributed.modelparallel.torch as smp
import torch


class DummyDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, length, data_type="openwebtext"):
        if data_type == "c4/en": 
            self.batch = (torch.Tensor(0), torch.Tensor(0))
        self.length = length

    def __getitem__(self, index):
        return self.batch

    def __len__(self):
        return self.length



###### Load Openwebtext pretraining data ######
class C4enPretrainingDataset(torch.utils.data.Dataset):
    def __init__(self, input_paths: List[str], max_sequence_length=None, zipped=False, use_last_file_only=False):
        self.input_paths = input_paths
        self.max_sequence_length = max_sequence_length
        self.zipped = zipped
        self.use_last_file_only = use_last_file_only

        self.__read_examples(self.input_paths)

    def __read_examples(self, paths: List[str]):

        self.input_data = []
        if self.zipped:
            if self.use_last_file_only:
                with gzip.open(paths[-1], "rt") as f:
                    self.input_data = [ln for _, ln in enumerate(f, 1)]
            else:
                for path in paths:
                    with gzip.open(path, "rt") as f:
                        self.input_data.extend([ln for _, ln in enumerate(f, 1)])
        else:
            if self.use_last_file_only:
                with open (paths[-1], "r") as f:
                    self.input_data = [ln for ln in f]
            else:
                for path in paths:
                    with open (path, "r") as f:
                        self.input_data.extend([ln for ln in f])

        # print(f'__Finished building pretraining dataset with {self.iids.shape[0]} rows__')

    def __len__(self) -> int:
        return len(self.input_data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        obj = json.loads(self.input_data[index])
        iids = torch.tensor(obj["input_ids"], dtype=torch.long)
        attns = torch.tensor(obj["attention_mask"], dtype=torch.long)
        self.actual_sequence_length = len(obj["input_ids"])

        if self.actual_sequence_length > self.max_sequence_length:
            s_idx = np.random.randint(0, self.actual_sequence_length - self.max_sequence_length)
            e_idx = s_idx + self.max_sequence_length
            iids = iids[s_idx:e_idx]
            attns = attns[s_idx:e_idx]
        return iids, attns


def create_pretraining_dataloader(
    input_paths: List[str],
    batch_size: int,
    max_sequence_length: int,
    seed: int,
    dp_rank: int,
    dp_size: int,
    shuffle: bool = False,
    zipped: bool = True,
    use_last_file_only: bool = False,
    data_type: str = "c4",
):
    if smp.pp_rank() == 0:
        if data_type == "c4":
            data = C4enPretrainingDataset(
                input_paths=input_paths, max_sequence_length=max_sequence_length, zipped=zipped, use_last_file_only=use_last_file_only
            )
        else:
            raise ValueError(f"Unsupported data type {data_type}")
        sampler = torch.utils.data.DistributedSampler(
            data,
            shuffle=shuffle,
            seed=seed,
            rank=dp_rank,
            num_replicas=dp_size,
            drop_last=True,
        )
        dataloader = torch.utils.data.DataLoader(
            data,
            sampler=sampler,
            batch_size=batch_size,
            pin_memory=True,
            drop_last=True,
        )
        smp.broadcast(len(dataloader), smp.PP_GROUP)
    else:
        data_len = smp.recv_from(0, smp.RankType.PP_RANK)
        dataset = DummyDataset(data_len * batch_size, data_type=data_type)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=True)

    return dataloader
