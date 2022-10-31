import argparse
import json
from pathlib import Path
import random
import numpy as np
import torch

from src.models import fined
from src.utils import data_utils, logging, metrics, model_utils

logger = logging.get_logger(__name__)


def parse_args(use_args=None):
    parser = argparse.ArgumentParser()

    # Name the experiment.
    parser.add_argument(
        "--name", type=str, default="exp1", help="A name for the job."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to write output files. Defaults to 'output/$name'.",
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed.")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[],
        help="Run with multiple random seeds and average the results.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help=(
            "Train/evaluate multiple adapters on all examples in parallel. "
            "Otherwise the adapter names should match up with dataset names, "
            "and each adapter will only see examples from the matching dataset."
        ),
    )
    parser.add_argument(
        "--multi_head", action="store_true", help="Use multiple heads."
    )

    # Models
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="The name of a model on HuggingFace model hub.",
    )
    parser.add_argument(
        "--load_from",
        type=str,
        default=None,
        help=(
            "A file or directory containing a model.pt file to use to "
            "initialize the model parameters."
        ),
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save checkpoints over the course of training.",
    )
    parser.add_argument(
        "--save_every_best",
        action="store_true",
        help=(
            "Keep every best checkpoint (the default is to only keep the most "
            "recent best checkpoint)."
        ),
    )
    parser.add_argument(
        "--save_epoch_best",
        action="store_true",
        help=(
            "Keep the best checkpoint from every epoch (the default is to only "
            "keep the most recent best checkpoint)."
        ),
    )
    parser.add_argument(
        "--delete_model_at_end",
        action="store_true",
        help="Delete all model checkpoints at the end of training.",
    )
    parser.add_argument(
        "--freeze_transformer",
        action="store_true",
        help="Freeze Transformer parameters.",
    )
    parser.add_argument(
        "--freeze_heads",
        action="store_true",
        help="Freeze classifier heads.",
    )

    # Data
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help=(
            "Directory containing datasets. Expected format: "
            "${data_dir}/{train,dev}/${dataset}.jsonl "
        ),
    )
    parser.add_argument(
        "--train_on",
        type=str,
        nargs="+",
        default=[],
        help="The name of one or more datasets in $data_dir/train/.",
    )
    parser.add_argument(
        "--eval_on",
        type=str,
        nargs="+",
        default=[],
        help="The name of one or more datasets in $data_dir/eval/.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="cache",
        help=(
            "Directory to cache preprocessed datasets. "
            "(Leave empty to not use cache.)"
        ),
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cache and preprocess the datasets again.",
    )
    parser.add_argument(
        "--max_train_examples",
        type=int,
        default=None,
        help="Maximum number of training examples.",
    )
    parser.add_argument(
        "--max_dev_examples",
        type=int,
        default=None,
        help="Maximum number of development examples.",
    )
    
    # Training
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=512,
        help="maximum flatten dialogue length",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=0,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help=(
            "Number of optimization steps. (Model will train for the longer "
            "of --steps and --epochs.)"
        ),
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=1024,
        help="Number of optimization steps between checkpoints",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=-1,
        help=(
            "If > 0, stop training after this many checkpoints without "
            "improvement."
        ),
    )
    parser.add_argument(
        "--criterion",
        type=str,
        default="loss",
        help="Criterion to use for early stopping (loss or acc).",
    )
    parser.add_argument(
        "--full_eval_after_training",
        action="store_true",
        help="Evaluate on the full development set after training.",
    )
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--eval_before_training", action="store_true")

    # Sampling schedule
    parser.add_argument(
        "--bucket_sampler",
        action="store_true",
        help="Batch similar-length examples (to speed up training)",
    )
    parser.add_argument(
        "--dynamic_sampling",
        action="store_true",
        help="Dynamic sampling schedule for multi-dataset training.",
    )
    parser.add_argument(
        "--dynamic_sampling_after",
        type=int,
        default=10000,
        help="Number of optimization steps before dynamic sampling.",
    )

    # Learning rate
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--adapter_learning_rate", type=float, default=1e-4)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--scheduler", default=None)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)

    # misc
    parser.add_argument("--notebook", action="store_true")

    args = parser.parse_args(args=use_args)

    if args.output_dir is None:
        args.output_dir = f"output/{args.name}"

    if not Path(args.output_dir).exists():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if not args.eval_on:
        args.eval_on = args.train_on

    if args.model_name_or_path is None:
        args.model_name_or_path = "t5-base" if args.seq2seq else "roberta-base"

    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_onrun_one(args, seed, multi_seed=False):
    logger.info(f"run one: s{seed}")
    set_seed(seed)
    args.seed = seed
    prev_output_dir = args.output_dir
    if multi_seed:
        args.output_dir = str(Path(args.output_dir) / f"s{seed}")
        if not Path(args.output_dir).exists():
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    trainer = fined.FineTrainer()

    tokenizer, model = trainer.initialize(args)
    model.to(model_utils.device())
    if args.load_from:
        paths = [args.load_from] if args.load_from else []
        trainer.load_from(args, paths, model)

    # Set seed again before loading data
    set_seed(args.seed)

    train_results = None
    if args.train_on and args.max_train_examples:
        train_datasets = []
        for dataset in args.train_on:
            train_dataset = data_utils.load_dialogue_dataset(
                dataset,
                "train",
                tokenizer,
                max_examples=args.max_train_examples,
                overwrite_cache=args.overwrite_cache,
                seed=args.seed,
                data_dir=args.data_dir,
                cache_dir=args.cache_dir,
                max_seq_len=args.max_seq_len
            )
            train_datasets.append(train_dataset)
        if len(train_datasets) > 1:
            train_dataset = data_utils.GenericDatasets(train_datasets)
        else:
            train_dataset = train_datasets[0]

        eval_datasets = {}
        for dataset in args.eval_on:
            eval_datasets[dataset] = data_utils.load_dialogue_dataset(
                dataset,
                "dev",
                tokenizer,
                max_examples=args.max_dev_examples,
                overwrite_cache=args.overwrite_cache,
                seed=args.seed,
                data_dir=args.data_dir,
                cache_dir=args.cache_dir,
                max_seq_len=args.max_seq_len
            )

        train_results = trainer.train(
            args, model, tokenizer, train_dataset, eval_datasets
        )

    eval_results = None
    if args.eval_on:
        if args.full_eval_after_training:
            logger.info(f"evaluating on full development set")
            logger.info(f"loading best checkpoint")
            if args.separate_adapter_checkpoints:
                load_from = [
                    Path(args.output_dir) / a for a in args.adapter_names
                ]
            else:
                load_from = [args.output_dir]
            model = trainer.load_from(args, load_from, model)
            max_examples = None
        else:
            max_examples = args.max_dev_examples
        eval_datasets = {}
        for dataset in args.eval_on:
            eval_dataset = data_utils.load_dialogue_dataset(
                dataset,
                "dev",
                tokenizer,
                max_examples=max_examples,
                overwrite_cache=args.overwrite_cache,
                seed=args.seed,
                data_dir=args.data_dir,
                cache_dir=args.cache_dir,
            )
            eval_datasets[dataset] = eval_dataset
        eval_results = trainer.evaluate(
            args, model, tokenizer, eval_datasets, ckp="end"
        )
        
    model_fns = [Path(args.output_dir) / "model.pt"]
    for model_fn in model_fns:
        if model_fn.exists() and args.delete_model_at_end:
            logger.info(f"deleting {model_fn}")
            model_fn.unlink()

    args.output_dir = prev_output_dir
    return train_results, eval_results


if __name__ == "__main__":
    args = parse_args()

    logging.initialize(args.output_dir)
    logger.info(f"logging to {args.output_dir}/")
    logger.info(f"args: {vars(args)}")
    with open(Path(args.output_dir) / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    seeds = args.seeds or [args.seed]
    if args.max_train_examples == 0 and args.few_shot:
        seeds = [args.seed]

    eval_results = []
    for seed in seeds:
        _, result = train_onrun_one(args, seed, multi_seed=len(seeds) > 1)
        if result:
            eval_results.append(result)
    if eval_results and len(args.seeds) > 0:
        avg_results = metrics.average_dicts(eval_results)
        logger.info(f"average results: {avg_results}")
        fn = Path(args.output_dir) / f"metrics.avg.json"
        logger.info(f"writing average results to {fn}")
        with open(fn, "w") as f:
            json.dump(avg_results, f, indent=2)
