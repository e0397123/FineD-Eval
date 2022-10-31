from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path
import pickle
import random
import codecs
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, Sampler, RandomSampler
from torch.nn.utils.rnn import pad_sequence

from src.utils import logging

logger = logging.get_logger(__name__)

def dialog_tokens():
    return ["</UTT>"]

def cache_fn(name, split, tokenizer, cache_dir="cache"):
    return (Path(cache_dir) / split / name).with_suffix(
        "." + tokenizer.__class__.__name__ + ".pkl"
    )

@dataclass
class DialogueExample:
    qid: str
    dataset: str
    dial_a: List[str]
    dial_b: List[str]
    dial_len: int
    label: int

@dataclass
class DialogueFeature:
    qid: str
    input_ids_1: List[int]
    input_masks_1: List[int]
    token_type_ids_1: List[int]
    input_ids_2: List[int]
    input_masks_2: List[int]
    token_type_ids_2: List[int]
    dial_len: int
    labels: float


def load_dialogue_examples(fn):
    label2id = {'original':0, 'random':1}
    path = Path(fn).with_suffix(".txt")
    logger.info(f"loading examples from {path}")
    dataset = path.stem
    with codecs.open(path, mode="r", encoding='utf-8') as f:
        lines = f.readlines()
    ds = [l.strip().split('\t') for l in lines]
    examples = []
    for i, d in enumerate(ds):
        a = d[0]
        d1 = d[1]
        d2 = d[2]
        d1_utterances = d1.split('|||')
        d2_utterances = d2.split('|||')
        dialogue_length = len(d1_utterances)
        label = label2id[a]
        examples.append(
            DialogueExample(
                qid="dial-{}-{}".format(dataset, i),
                dataset=dataset,
                dial_a=d1_utterances,
                dial_b=d2_utterances,
                dial_len=dialogue_length,
                label=label
            )
        )
    return [asdict(e) for e in tqdm(examples)]


def _truncate_seq_front(tokens, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens)
        if total_length <= max_length:
            break
        else:
            tokens.pop(1)


def convert_dialogue_examples_to_features(
    examples,
    tokenizer,
    max_length=512
):
    logger.info(f"tokenizing {len(examples)} examples")
    features = []
    skip_count = 0
    for jdx, e in enumerate(tqdm(examples)):
        try:
            d_1 = e['dial_a']
            d_2 = e['dial_b']
            d_len = e['dial_len']
            dial_text_list_1 = []
            for idx, x in enumerate(d_1):
                dial_text_list_1.append(x)
            flatten_text_1 = '</UTT>'.join(dial_text_list_1) + '</UTT>'
            input_ids_1 = tokenizer.encode(flatten_text_1)
            _truncate_seq_front(input_ids_1, max_length)
            input_masks_1 = [1] * len(input_ids_1)
            token_type_ids_1 = [0] * len(input_ids_1)
            while len(input_ids_1) < max_length:
                input_ids_1.append(tokenizer.pad_token_id)
                input_masks_1.append(0)
                token_type_ids_1.append(0)

            dial_text_list_2 = []
            for idx, x in enumerate(d_2):
                dial_text_list_2.append(x)
            flatten_text_2 = '</UTT>'.join(dial_text_list_2) + '</UTT>'
            input_ids_2 = tokenizer.encode(flatten_text_2)
            _truncate_seq_front(input_ids_2, max_length)
            input_masks_2 = [1] * len(input_ids_2)
            token_type_ids_2 = [0] * len(input_ids_2)
            while len(input_ids_2) < max_length:
                input_ids_2.append(tokenizer.pad_token_id)
                input_masks_2.append(0)
                token_type_ids_2.append(0)
        except Exception as ex:
            logger.warning(f"error tokenizing example {idx}: {ex}")
            skip_count += 1
            continue

        features.append(DialogueFeature("dial-{}-{}".format(e['dataset'], jdx),
                                        input_ids_1,
                                        input_masks_1,
                                        token_type_ids_1,
                                        input_ids_2,
                                        input_masks_2,
                                        token_type_ids_2,
                                        d_len,
                                        e['label']))
    logger.info(
        f"converted {len(examples) - skip_count}/{len(examples)} examples "
        f"into {len(features)} features"
    )
    return [asdict(f) for f in tqdm(features)]


def load_dialogue_dataset(
    name,
    split,
    tokenizer,
    max_examples=None,
    overwrite_cache=False,
    seed=13,
    data_dir="data",
    cache_dir="cache",
    max_seq_len=512
):
    fn = Path(data_dir) / split / name
    cache = cache_fn(name, split, tokenizer)
    if cache_dir and cache.exists() and not overwrite_cache:
        logger.info(f"loading cached dataset from {cache}")
        with open(cache, "rb") as f:
            examples, features = pickle.load(f)
    else:
        examples = load_dialogue_examples(fn)
        features = convert_dialogue_examples_to_features(examples,
                                                         tokenizer,
                                                         max_length=max_seq_len)
        if cache_dir:
            logger.info(f"caching dataset to {cache}")
            cache.parent.mkdir(exist_ok=True, parents=True)
            with open(cache, "wb") as f:
                pickle.dump((examples, features), f)

    if (max_examples or 0) > 0 and max_examples < len(examples):
        random.seed(seed)
        examples = random.sample(examples, max_examples)
        qids = set(e["qid"] for e in examples)
        features = [f for f in features if f["qid"] in qids]
        logger.info(
            f"picked {len(examples)} examples / {len(features)} features"
        )

    dataset = GenericDialogueDataset(examples, features)
    return dataset


def collate(batch):
    examples, features = zip(*batch)
    d = {"examples": examples}
    d.update(collate_(features))
    return d


def collate_(features):
    d = {}
    for k in features[0].keys():
        values = [f[k] for f in features]
        if type(values[0]) == str:
            d[k] = values
        elif type(values[0]) in (int, float):
            d[k] = torch.tensor(values)
        elif type(values[0]) == list:
            if type(values[0][0]) != list:
                max_len = max(len(v) for v in values)
                mask_k = "attention_mask" if k == "input_ids" else f"{k}_mask"
                d[k] = torch.full((len(features), max_len), 0, dtype=torch.long)
                d[mask_k] = torch.full(
                    (len(features), max_len), 0, dtype=torch.bool
                )
                for i, v in enumerate(values):
                    t = torch.tensor(v, dtype=torch.long)
                    m = torch.ones_like(t, dtype=torch.bool)
                    d[k][i, : len(t)] = t
                    d[mask_k][i, : len(m)] = m
            else:
                max_len = max(len(v) for v in values)
                seq_len = len(values[0][0])
                mask_k = "attention_mask" if k == "input_ids" else f"{k}_mask"
                d[k] = torch.full((len(features), max_len, seq_len), 0, dtype=torch.long)
                d[mask_k] = torch.full(
                    (len(features), max_len, seq_len), 0, dtype=torch.bool
                )
                for i, v in enumerate(values):
                    for j, item in enumerate(v):
                        t = torch.tensor(item, dtype=torch.long)
                        m = torch.ones_like(t, dtype=torch.bool)
                        d[k][i, j, : len(t)] = t
                        d[mask_k][i, j, : len(m)] = m

        elif type(values[0]) == dict:
            d[k] = collate_(values)
        elif type(values[0]) == torch.Tensor:
            d[k] = pad_sequence(values, batch_first=True)
        else:
            raise NotImplementedError(type(values[0]))
    return d


class GenericDialogueDataset:
    def __init__(self, examples, features):
        self.name = examples[0]["dataset"]
        self.examples = examples
        self.features = features
        self.qid_to_e = {e["qid"]: e for e in examples}

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        f = self.features[idx]
        e = self.qid_to_e[f["qid"]]
        return e, f

    def split(self):
        qids = list(self.qid_to_e.keys())
        random.shuffle(qids)
        a_qids = set(qids[: len(qids) // 2])
        a_features = [f for f in self.features if f["qid"] in a_qids]
        a_examples = [e for e in self.examples if e["qid"] in a_qids]
        b_features = [f for f in self.features if f["qid"] not in a_qids]
        b_examples = [e for e in self.examples if e["qid"] not in a_qids]
        return GenericDialogueDataset(a_examples, a_features), GenericDialogueDataset(
            b_examples, b_features
        )


class GenericDatasets:
    def __init__(self, datasets):
        self.datasets = datasets
        self.offsets = [0]
        for d in datasets:
            self.offsets.append(self.offsets[-1] + len(d))

    @property
    def examples(self):
        return (e for d in self.datasets for e in d.examples)

    @property
    def features(self):
        return (f for d in self.datasets for f in d.features)

    def __len__(self):
        return self.offsets[-1]

    def __getitem__(self, idx):
        for i in range(len(self.datasets)):
            if idx < self.offsets[i + 1]:
                return self.datasets[i][idx - self.offsets[i]]
        raise ValueError(idx)

def dynamic_sampling_weights():
    return {
        "dailydialog": 0.906 + 0.836,
        "convai2": 0.797 + 0.664,
        "empathetic": 0.779 + 0.734,
        "topical": 0.551 + 0.724,
        "reddit": 0.847 + 0.796,
    }


class WeightedSampler(Sampler):
    def __init__(self, dataset, batch_size, num_buckets=1):
        assert hasattr(dataset, "datasets")
        self.dataset_names = [d.name for d in dataset.datasets]
        self.offsets = dict(zip(self.dataset_names, dataset.offsets))
        self.length = len(dataset)
        self.batch_size = batch_size
        self.num_buckets = num_buckets
        self.weights = [1 for _ in range(len(self.dataset_names))]

    def set_weights(self, weights):
        self.weights = [weights[t] for t in self.train_on]

    def set_dynamic_sampling_weights(self, report):
        baseline = dynamic_sampling_weights()
        diffs = []
        for t in self.dataset_names:
            cur = report[t]["em"] + report[t]["f1"]
            diffs.append(abs(baseline[t] - cur))
        new_weights = [d / sum(diffs) for d in diffs]
        logger.info(f"adjusting dynamic sampling weights")
        logger.info(f"current: {dict(zip(self.dataset_names, self.weights))}")
        logger.info(f"new: {dict(zip(self.dataset_names, new_weights))}")
        self.weights = new_weights

    def current_weights(self):
        return dict(zip(self.dataset_names, self.weights))

    def __iter__(self):
        raise NotImplementedError

    def __len__(self):
        return self.length // self.batch_size


class DatasetMixerSampler(WeightedSampler):
    def __init__(self, dataset, batch_size, sort_datasets=True, num_buckets=1):
        super().__init__(dataset, batch_size, num_buckets)
        self.subsamplers = {
            subdataset.name: RandomSampler(subdataset)
            for subdataset in dataset.datasets
        }
        self.iterators = {
            name: iter(subsampler)
            for name, subsampler in self.subsamplers.items()
        }
        self.sort_datasets = sort_datasets

    def subsample(self, dataset_name):
        e = next(self.iterators[dataset_name], None)
        if e is None:
            self.iterators[dataset_name] = iter(self.subsamplers[dataset_name])
            e = next(self.iterators[dataset_name])
        return e + self.offsets[dataset_name]

    def __iter__(self):
        for _ in range(self.length // self.batch_size):
            datasets = random.choices(
                self.dataset_names, weights=self.weights, k=self.batch_size
            )
            if self.sort_datasets:
                datasets = sorted(datasets)
            batch = [self.subsample(d) for d in datasets]
            yield batch


class BucketSampler(Sampler):
    def __init__(self, dataset, batch_size, num_buckets=8, offset=0):
        lens = [len(f["input_ids_1"]) for f in dataset.features]
        self.idxs = idxs = [
            offset + i for i in sorted(range(len(lens)), key=lambda i: lens[i])
        ]
        bucket_size = round(len(idxs) / num_buckets)
        self.buckets = [
            idxs[i : i + bucket_size] for i in range(0, len(idxs), bucket_size)
        ]
        self.batch_size = batch_size

    def __iter__(self):
        for bucket in self.buckets:
            random.shuffle(bucket)
        batches = []
        batch = []
        for i in (i for b in self.buckets for i in b):
            batch.append(i)
            if len(batch) == self.batch_size:
                batches.append(batch)
                batch = []
        if batch:
            batches.append(batch)
        random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return round(len(self.idxs) / self.batch_size)
