#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
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
Fine-tuning the library models for question answering integrated with sparseml
"""
import json
import logging
import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Optional

import datasets
import transformers
from datasets import load_dataset, load_metric
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.versions import require_version

from sparseml.pytorch.utils.distributed import record
from sparseml.transformers.sparsification import (
    QuestionAnsweringTrainer,
    TrainingArguments,
    postprocess_qa_predictions,
)
from sparseml.transformers.utils import SparseAutoModel, get_shared_tokenizer_src


# You can also adapt this script on your own question answering task.
# Pointers for this are left as comments.


require_version(
    "datasets>=1.18.0",
    "To fix: pip install -r examples/pytorch/question-answering/requirements.txt",
)


_LOGGER = logging.getLogger(__name__)

metadata_args = [
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "doc_stride",
    "fp16",
    "max_seq_length",
]


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from
    """

    model_name_or_path: str = field(
        metadata={
            "help": (
                "Path to pretrained model or model identifier from "
                "huggingface.co/models"
            )
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name",
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name",
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Path to directory to store the pretrained models downloaded from "
                "huggingface.co"
            ),
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": (
                "The specific model version to use (can be a branch name, "
                "tag name or commit id)."
            ),
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` "
                "(necessary to use this script with private models)."
            ),
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data to input to our model for training and eval
    """

    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the dataset to use (via the datasets library).",
        },
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The configuration name of the dataset to use "
                "(via the datasets library)."
            ),
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a text file)."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the perplexity "
                "on (a text file)."
            ),
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input test data file to evaluate the perplexity on "
                "(a text file)."
            ),
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. "
            "Sequences longer  than this will be truncated, sequences shorter will "
            "be padded."
        },
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
    version_2_with_negative: bool = field(
        default=False,
        metadata={"help": "If true, some of the examples do not have an answer."},
    )
    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={
            "help": (
                "The threshold used to select the null answer: if the best answer has "
                "a score that is less than the score of the null answer minus this "
                "threshold, the null answer is selected for this example. Only useful "
                "when `version_2_with_negative=True`."
            ),
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={
            "help": (
                "When splitting up a long document into chunks, how much stride to "
                "take between chunks."
            ),
        },
    )
    n_best_size: int = field(
        default=20,
        metadata={
            "help": (
                "The total number of n-best predictions to generate when looking "
                "for an answer."
            ),
        },
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": (
                "The maximum length of an answer that can be generated. This is "
                "needed because the start and end predictions are not conditioned "
                "on one another."
            ),
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
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and self.test_file is None
        ):
            raise ValueError(
                "Need either a dataset name or a training/validation file/test_file."
            )
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                ], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                ], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                ], "`test_file` should be a csv or a json file."


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
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is "
                "not empty. Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            _LOGGER.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. "
                "To avoid this behavior, change the `--output_dir` or add "
                "`--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can
    # concurrently download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model, teacher = SparseAutoModel.question_answering_from_pretrained_distil(
        model_name_or_path=model_args.model_name_or_path,
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

    tokenizer_src = (
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else get_shared_tokenizer_src(model, teacher)
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_src,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Tokenizer check: this script requires a fast tokenizer.
    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        raise ValueError(
            "This example script only works for models that have a fast tokenizer. "
            "Checkout the big table of models at "
            "https://huggingface.co/transformers/index.html#supported-frameworks to "
            "find the model types that meet this requirement"
        )

    raw_datasets = _get_raw_dataset(data_args=data_args, cache_dir=model_args.cache_dir)
    make_eval_dataset = training_args.do_eval or data_args.num_export_samples > 0
    tokenized_datasets, examples = _get_tokenized_datasets_and_examples(
        data_args=data_args,
        raw_datasets=raw_datasets,
        tokenizer=tokenizer,
        make_eval_dataset=make_eval_dataset,
        do_train=training_args.do_train,
        do_predict=training_args.do_predict,
        main_process_func=training_args.main_process_first,
    )

    train_dataset = tokenized_datasets.get("train")
    eval_dataset, eval_examples = tokenized_datasets.get("validation"), examples.get(
        "validation"
    )
    predict_dataset, predict_examples = tokenized_datasets.get("test"), examples.get(
        "test"
    )

    # Data collator
    # We have already padded to max length if the corresponding flag is True,
    # otherwise we need to pad in the data collator.
    data_collator = (
        default_data_collator
        if data_args.pad_to_max_length
        else DataCollatorWithPadding(
            tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
        )
    )
    column_names = _get_column_names(
        raw_datasets=raw_datasets,
        make_eval_dataset=make_eval_dataset,
        do_train=training_args.do_train,
    )
    answer_column_name = "answers" if "answers" in column_names else column_names[2]

    # Post-processing:
    def post_processing_function(examples, features, predictions, stage="eval"):
        # Post-processing: we match the start logits and end
        # logits to answers in the original context.
        predictions = postprocess_qa_predictions(
            examples=examples,
            features=features,
            predictions=predictions,
            version_2_with_negative=data_args.version_2_with_negative,
            n_best_size=data_args.n_best_size,
            max_answer_length=data_args.max_answer_length,
            null_score_diff_threshold=data_args.null_score_diff_threshold,
            output_dir=training_args.output_dir,
            log_level=log_level,
            prefix=stage,
        )
        # Format the result to the format the metric expects.
        if data_args.version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0}
                for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [
                {"id": k, "prediction_text": v} for k, v in predictions.items()
            ]

        references = [
            {"id": ex["id"], "answers": ex[answer_column_name]} for ex in examples
        ]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    metric = load_metric("squad_v2" if data_args.version_2_with_negative else "squad")

    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    # Initialize our Trainer
    trainer = QuestionAnsweringTrainer(
        model=model,
        model_state_path=model_args.model_name_or_path,
        recipe=training_args.recipe,
        recipe_args=training_args.recipe_args,
        metadata_args=metadata_args,
        teacher=teacher,
        args=training_args,
        data_args=data_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if make_eval_dataset else None,
        eval_examples=eval_examples if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
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

