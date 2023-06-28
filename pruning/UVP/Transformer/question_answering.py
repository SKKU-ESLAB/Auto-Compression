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


