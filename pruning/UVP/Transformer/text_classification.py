#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
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

# Adapted from https://github.com/huggingface/transformers
# neuralmagic: no copyright

"""
Finetuning the library models for sequence classification on GLUE
"""

# You can also adapt this script on your own text classification task.
# Pointers for this are left as comments.

import logging
import os
import random
import sys
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Optional

import datasets
import numpy as np
import transformers
from datasets import load_dataset, load_metric
from sklearn.model_selection import StratifiedShuffleSplit
from torch.nn import Module
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version

from sparseml.pytorch.utils.distributed import record
from sparseml.transformers.sparsification import Trainer, TrainingArguments
from sparseml.transformers.utils import (
    SparseAutoModel,
    get_shared_tokenizer_src,
    multi_label_precision_recall_f1,
)


require_version(
    "datasets>=1.18.0",
    "To fix: pip install -r examples/pytorch/text-classification/requirements.txt",
)

_TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "imdb": ("text", None),
}

_LOGGER: logging.Logger = logging.getLogger(__name__)

metadata_args = [
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "fp16",
]


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for
    training and eval

    Using `HfArgumentParser` we can turn this class into argparse
    arguments to be able to specify them on the command line
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the task to train on: "
            + ", ".join(_TASK_TO_KEYS.keys())
        },
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)"},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": ("The configuration name of the dataset to use"),
        },
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. "
            "Sequences longer  than this will be truncated, sequences shorter will "
            "be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. If False, "
            "will pad the samples dynamically when batching to the maximum length "
            "in the batch (which can be faster on GPU but will be slower on TPU)."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number "
            "of training examples to this value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number "
            "of evaluation examples to this value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of "
                "prediction examples to this value if set."
            ),
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the training data."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the validation data."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the test data."},
    )
    validation_ratio: Optional[float] = field(
        default=None,
        metadata={"help": "Percentage of the training data to be used as validation."},
    )
    eval_on_test: bool = field(
        default=False,
        metadata={"help": "Evaluate the test dataset."},
    )
    input_column_names: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "name of column to read model input data from. May also be comma "
                "separated list of two columns to use as inputs. Examples include "
                "'sentence' for single column and 'sentence_1,sentence_2' for two. "
                "Default behavior is to read columns based on task name or infer from "
                "non 'label' columns if sentence_column_names and task name not"
                "provided"
            )
        },
    )
    label_column_name: str = field(
        default="label",
        metadata={
            "help": (
                "column in dataset where input labels are located. Default is 'label'"
            )
        },
    )
    one_shot: bool = field(
        default=False,
        metadata={"help": "Whether to apply recipe in a one shot manner."},
    )
    num_export_samples: int = field(
        default=0,
        metadata={"help": "Number of samples (inputs/outputs) to export during eval."},
    )

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            if self.task_name not in _TASK_TO_KEYS.keys():
                raise ValueError(
                    "Unknown task, you should pick one in "
                    + ",".join(_TASK_TO_KEYS.keys())
                )
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError(
                "Need either a GLUE task, a training/validation file or a dataset name"
            )
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in [
                "csv",
                "json",
            ], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert validation_extension == train_extension, (
                "`validation_file` should have the same extension (csv or json) "
                "as `train_file`."
            )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from
    """

    model_name_or_path: str = field(
        metadata={
            "help": (
                "Path to pretrained model, sparsezoo stub. or model identifier from "
                "huggingface.co/models"
            )
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained data from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizers. Default True"},
    )
    use_teacher_tokenizer: bool = field(
        default=False,
        metadata={
            "help": "Whether to use separate tokenizer for distillation teacher. "
            "Default False; uses same tokenizer for teacher and student"
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use "
            "(can be a branch name, tag name or commit id)"
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use token generated when running `transformers-cli login` "
            "(necessary to use this script with private models)"
        },
    )


@record
def main(**kwargs):
    # See all possible arguments in
    # src/sparseml/transformers/sparsification/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    elif not kwargs:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    else:
        model_args, data_args, training_args = parser.parse_dict(kwargs)
    # Setup logging

    log_level = training_args.get_process_log_level()
    _LOGGER.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    _LOGGER.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, "
        f"distributed training: {bool(training_args.local_rank != -1)}, "
        f"16-bits training: {training_args.fp16}"
    )
    _LOGGER.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and (len(os.listdir(training_args.output_dir)) > 0):
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already "
                "exists and is not empty. Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            _LOGGER.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To "
                "avoid this behavior, change  the `--output_dir` or add "
                "`--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    raw_datasets = _get_raw_dataset(
        data_args, cache_dir=model_args.cache_dir, do_predict=training_args.do_predict
    )

    # Labels
    (
        is_regression,
        label_column,
        label_list,
        num_labels,
        is_multi_label_classification,
    ) = _get_label_info(data_args, raw_datasets)

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one
    # local process can concurrently download model & vocab.
    config_kwargs = {}
    if is_multi_label_classification:
        config_kwargs["problem_type"] = "multi_label_classification"
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        **config_kwargs,
    )

    model, teacher = SparseAutoModel.text_classification_from_pretrained_distil(
        model_name_or_path=(
            model_args.tokenizer_name
            if model_args.tokenizer_name
            else model_args.model_name_or_path
        ),
        model_kwargs={
            "config": config,
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "use_auth_token": True if model_args.use_auth_token else None,
        },
        teacher_name_or_path=training_args.distill_teacher,
        teacher_kwargs={
            "cache_dir": model_args.cache_dir,
            "use_auth_token": True if model_args.use_auth_token else None,
        },
    )

    teacher_tokenizer = None
    tokenizer_kwargs = dict(
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    if not model_args.use_teacher_tokenizer:
        tokenizer_src = (
            model_args.tokenizer_name
            if model_args.tokenizer_name
            else get_shared_tokenizer_src(model, teacher)
        )
    else:
        tokenizer_src = (
            model_args.tokenizer_name
            if model_args.tokenizer_name
            else model.config._name_or_path
        )
        teacher_tokenizer = AutoTokenizer.from_pretrained(
            teacher.config._name_or_path,
            **tokenizer_kwargs,
        )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_src,
        **tokenizer_kwargs,
    )
    make_eval_dataset = training_args.do_eval or data_args.num_export_samples > 0
    tokenized_datasets, raw_datasets = _get_tokenized_and_preprocessed_raw_datasets(
        config=config,
        data_args=data_args,
        model=model,
        raw_datasets=raw_datasets,
        tokenizer=tokenizer,
        teacher_tokenizer=teacher_tokenizer,
        make_eval_dataset=make_eval_dataset,
        main_process_func=training_args.main_process_first,
        do_train=training_args.do_train,
        do_predict=training_args.do_predict,
    )

    train_dataset = tokenized_datasets.get("train")
    eval_dataset = tokenized_datasets.get("validation")
    predict_dataset = tokenized_datasets.get("test")

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            _LOGGER.info(f"Sample {index} of training set: {train_dataset[index]}.")

    # Get the metric function
    if data_args.task_name is not None:
        metric = load_metric("glue", data_args.task_name)
    else:
        metric = load_metric("accuracy")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction`
    # object (a namedtuple with a predictions and label_ids field) and has to return a
    # dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        if is_regression:
            preds = np.squeeze(preds)
        elif not is_multi_label_classification:
            # do not run argmax for multi label classification
            preds = np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        elif is_multi_label_classification:
            threshold = 0.3  # from go_emotions paper - potentially move to arg/config
            preds_sigmoid = 1 / (1 + np.exp(-preds))
            multi_label_preds = (preds_sigmoid > threshold).astype(np.float32)
            label_to_id = _get_label_to_id(
                data_args=data_args,
                is_regression=is_regression,
                label_list=label_list,
                model=model,
                num_labels=num_labels,
                config=config,
            )
            id_to_label = {id_: label for label, id_ in label_to_id.items()}

            return multi_label_precision_recall_f1(
                predictions=multi_label_preds,
                targets=p.label_ids,
                id_to_label=id_to_label,
            )
        else:
            return {
                "accuracy": (preds == p.label_ids).astype(np.float32).mean().item(),
            }

    # Data collator will default to DataCollatorWithPadding when the tokenizer is
    # passed to Trainer, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        model_state_path=model_args.model_name_or_path,
        recipe=training_args.recipe,
        metadata_args=metadata_args,
        recipe_args=training_args.recipe_args,
        teacher=teacher,
        args=training_args,
        data_args=data_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if make_eval_dataset else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        if not trainer.one_shot:
            metrics = train_result.metrics
            max_train_samples = (
                data_args.max_train_samples
                if data_args.max_train_samples is not None
                else len(train_dataset)
            )
            metrics["train_samples"] = min(max_train_samples, len(train_dataset))
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)

        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.save_state()
        trainer.save_optimizer_and_scheduler(training_args.output_dir)

    # Evaluation
    if training_args.do_eval and not trainer.one_shot:
        _LOGGER.info("*** Evaluate ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = (
            [data_args.task_name]
            if data_args.task_name is not None
            else [data_args.dataset_name]
        )
        eval_datasets = [eval_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            eval_datasets.append(raw_datasets["validation_mismatched"])
            combined = {}

        for eval_dataset, task in zip(eval_datasets, tasks):
            metrics = trainer.evaluate(eval_dataset=eval_dataset)

            max_eval_samples = (
                data_args.max_eval_samples
                if data_args.max_eval_samples is not None
                else len(eval_dataset)
            )
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
            if task == "mnli-mm":
                metrics = {k + "_mm": v for k, v in metrics.items()}
            if task is not None and "mnli" in task:
                combined.update(metrics)
                trainer.save_metrics("eval", combined)
            else:
                trainer.save_metrics("eval", metrics)

            trainer.log_metrics("eval", metrics)

    if training_args.do_predict and not trainer.one_shot:
        _LOGGER.info("*** Predict ***")

        # Loop to handle MNLI double evaluation (matched, mis-matched)
        tasks = [data_args.task_name]
        predict_datasets = [predict_dataset]
        if data_args.task_name == "mnli":
            tasks.append("mnli-mm")
            predict_datasets.append(raw_datasets["test_mismatched"])

        for predict_dataset, task in zip(predict_datasets, tasks):
            # Removing the `label` columns because it contains -1 and Trainer will
            # not like that
            predict_dataset = predict_dataset.remove_columns(label_column)
            predictions = trainer.predict(
                predict_dataset, metric_key_prefix="predict"
            ).predictions
            predictions = (
                np.squeeze(predictions)
                if is_regression
                else np.argmax(predictions, axis=1)
            )

            output_predict_file = os.path.join(
                training_args.output_dir, f"predict_results_{task}.txt"
            )
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    _LOGGER.info(f"***** Predict results {task} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            item = label_list[item]
                            writer.write(f"{index}\t{item}\n")

    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "text-classification",
    }
    if data_args.task_name is not None:
        kwargs["language"] = "en"
        kwargs["dataset_tags"] = "glue"
        kwargs["dataset_args"] = data_args.task_name
        kwargs["dataset"] = f"GLUE {data_args.task_name.upper()}"

    # Exporting Samples

    if data_args.num_export_samples > 0:
        trainer.save_sample_inputs_outputs(
            num_samples_to_export=data_args.num_export_samples
        )


def _get_label_info(data_args, raw_datasets):
    label_column = data_args.label_column_name
    if data_args.task_name is not None:
        is_regression = data_args.task_name == "stsb"
        is_multi_label_classification = False
        if not is_regression:
            label_list = raw_datasets["train"].features[label_column].names
            num_labels = len(label_list)
        else:
            num_labels = 1
    else:
        # Trying to have good defaults here, don't hesitate to tweak to your needs.
        label_type = raw_datasets["train"].features[label_column].dtype
        is_regression = label_type in ["float32", "float64"]
        is_multi_label_classification = label_type == "list"

        if is_regression:
            num_labels = 1
        elif is_multi_label_classification:
            label_list = raw_datasets["train"].features[label_column].feature.names
            num_labels = len(label_list)
        else:
            label_list = raw_datasets["train"].unique(label_column)
            label_list.sort()  # Let's sort it for determinism
            num_labels = len(label_list)
    return (
        is_regression,
        label_column,
        label_list,
        num_labels,
        is_multi_label_classification,
    )


def _get_tokenized_and_preprocessed_raw_datasets(
    config,
    data_args: DataTrainingArguments,
    model: Optional[Module],
    raw_datasets,
    tokenizer: transformers.PreTrainedTokenizerBase,
    teacher_tokenizer=None,
    make_eval_dataset: bool = False,
    do_predict: bool = False,
    do_train: bool = False,
    main_process_func=None,
):
    (
        is_regression,
        label_column,
        label_list,
        num_labels,
        is_multi_label_classification,
    ) = _get_label_info(data_args, raw_datasets)

    train_dataset = predict_dataset = eval_dataset = None
    config = model.config if model else config
    if not main_process_func:
        main_process_func = lambda desc: nullcontext(desc)  # noqa: E731

    # Preprocessing the datasets
    if data_args.input_column_names is not None:
        if "," in data_args.input_column_names:
            # two input columns
            columns = data_args.input_column_names.split(",")
            if len(columns) != 2:
                raise ValueError(
                    "input_column_names may only specify up to two columns "
                    f"{len(columns)} provided: {columns}"
                )
            sentence1_key, sentence2_key = columns
        else:
            # one input column
            sentence1_key = data_args.input_column_names
            sentence2_key = None
    elif data_args.task_name is not None:
        sentence1_key, sentence2_key = _TASK_TO_KEYS[data_args.task_name]
    else:
        # Again, we try to have some nice defaults but don't hesitate to tweak to your
        # use case
        non_label_column_names = [
            name for name in raw_datasets["train"].column_names if name != label_column
        ]
        if (
            "sentence1" in non_label_column_names
            and "sentence2" in non_label_column_names
        ):
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence
        # length in each batch
        padding = False

    # Some models have set the order of the labels to use, so let's make sure
    # we do use it
    label_to_id = _get_label_to_id(
        data_args, is_regression, label_list, model, num_labels, config=config
    )

    if label_to_id is not None:
        config.label2id = label_to_id
        config.id2label = {id: label for label, id in config.label2id.items()}
    elif data_args.task_name is not None and not is_regression:
        config.label2id = {l: i for i, l in enumerate(label_list)}
        config.id2label = {id: label for label, id in config.label2id.items()}

    max_seq_length = data_args.max_seq_length
    if max_seq_length > tokenizer.model_max_length:
        _LOGGER.warning(
            f"The max_seq_length passed ({max_seq_length}) is larger than "
            f"the maximum length for the model ({tokenizer.model_max_length}). "
            f"Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
