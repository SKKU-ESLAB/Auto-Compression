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


