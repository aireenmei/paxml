# coding=utf-8
# Copyright 2022 Google LLC.
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
"""Configs for benchmarking decoder-only models on C4 dataset."""

from absl import logging
import jax
from jax import numpy as jnp
from paxml import base_experiment
from paxml import experiment_registry
from paxml import tasks_lib
from paxml.tasks.lm import model_params
from paxml.tasks.lm.params import lm_cloud
from paxml.tasks.lm.params import c4
from praxis import layers

@experiment_registry.register
class C4Spmd2BAdam4Replicas(c4.C4SpmdAdam):
  r"""GPT-3 config with 16B params. Model Parameters: 
  Global batch size = 1 * 4 * 1 * 32 = 128"""
  NUM_LAYERS = 18
  MODEL_DIMS = 3072
  HIDDEN_DIMS = MODEL_DIMS * 4
  NUM_HEADS = 24
  DIMS_PER_HEAD = 128
  PERCORE_BATCH_SIZE = 32
  MAX_SEQ_LEN = 1024
  VOCAB_SIZE = 32000
  FPROP_DTYPE = jnp.bfloat16
  USE_REPEATED_LAYER = True

  SUMMARY_INTERVAL_STEPS = 10
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  ICI_MESH_SHAPE = [1, 4, 1]

@experiment_registry.register
class C4Spmd16BAdam32Replicas(c4.C4SpmdAdam):
  r"""GPT-3 config with 16B params. Model Parameters: 
  Global batch size = 1 * 2 * 16 * 16 = 512"""
  NUM_LAYERS = 36
  MODEL_DIMS = 6144
  HIDDEN_DIMS = MODEL_DIMS * 4
  NUM_HEADS = 48
  DIMS_PER_HEAD = 128
  PERCORE_BATCH_SIZE = 16
  MAX_SEQ_LEN = 1024
  VOCAB_SIZE = 32000
  FPROP_DTYPE = jnp.bfloat16
  USE_REPEATED_LAYER = True

  SUMMARY_INTERVAL_STEPS = 10
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  ICI_MESH_SHAPE = [1, 16, 2]

@experiment_registry.register
class C4Spmd16BAdam16Replicas(c4.C4SpmdAdam):
  r"""GPT-3 config with 16B params. Model Parameters: 
  Global batch size = 1 * 1 * 16 * 8 = 128"""
  NUM_LAYERS = 36
  MODEL_DIMS = 6144
  HIDDEN_DIMS = MODEL_DIMS * 4
  NUM_HEADS = 48
  DIMS_PER_HEAD = 128
  PERCORE_BATCH_SIZE = 8
  MAX_SEQ_LEN = 1024
  VOCAB_SIZE = 32000
  FPROP_DTYPE = jnp.bfloat16
  USE_REPEATED_LAYER = True

  SUMMARY_INTERVAL_STEPS = 10
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  ICI_MESH_SHAPE = [1, 16, 1]

@experiment_registry.register
class C4Spmd32BAdam64Replicas(c4.C4SpmdAdam):
  r"""GPT-3 config with 32B params. Model Parameters: 
  Global batch size = 1 * 16 * 4 * 8 = 512"""
  NUM_LAYERS = 40
  MODEL_DIMS = 8192
  HIDDEN_DIMS = MODEL_DIMS * 4
  NUM_HEADS = 64
  DIMS_PER_HEAD = 128
  PERCORE_BATCH_SIZE = 8
  MAX_SEQ_LEN = 1024
  VOCAB_SIZE = 32000
  FPROP_DTYPE = jnp.bfloat16
  USE_REPEATED_LAYER = True

  SUMMARY_INTERVAL_STEPS = 10
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  ICI_MESH_SHAPE = [1, 16, 4]

@experiment_registry.register
class C4Spmd64BAdam128Replicas(c4.C4SpmdAdam):
  r"""GPT-3 config with 64B params. Model Parameters: 
  Global batch size = 1 * 16 * 8 * 8 = 1024"""
  NUM_LAYERS = 51
  MODEL_DIMS = 10240
  HIDDEN_DIMS = MODEL_DIMS * 4
  NUM_HEADS = 80
  DIMS_PER_HEAD = 128
  PERCORE_BATCH_SIZE = 8
  MAX_SEQ_LEN = 1024
  VOCAB_SIZE = 32000
  FPROP_DTYPE = jnp.bfloat16
  USE_REPEATED_LAYER = True

  SUMMARY_INTERVAL_STEPS = 10
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  ICI_MESH_SHAPE = [1, 16, 8]

@experiment_registry.register
class C4Spmd128BAdam256Replicas(c4.C4SpmdAdam):
  r"""GPT-3 config with 128B params. Model Parameters: 
  Global batch size = 1 * 64 * 4 * 8 = 1024"""
  NUM_LAYERS = 71
  MODEL_DIMS = 12288
  HIDDEN_DIMS = MODEL_DIMS * 4
  NUM_HEADS = 96
  DIMS_PER_HEAD = 128
  PERCORE_BATCH_SIZE = 4
  MAX_SEQ_LEN = 1024
  VOCAB_SIZE = 32000
  FPROP_DTYPE = jnp.bfloat16
  USE_REPEATED_LAYER = True

  SUMMARY_INTERVAL_STEPS = 10
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  ICI_MESH_SHAPE = [1, 64, 4]

@experiment_registry.register
class C4Spmd256BAdam512Replicas(c4.C4SpmdAdam):
  r"""GPT-3 config with 256B params. Model Parameters: 
  Global batch size = 1 * 64 * 8 * 4 = 2048"""
  NUM_LAYERS = 80
  MODEL_DIMS = 16384
  HIDDEN_DIMS = MODEL_DIMS * 4
  NUM_HEADS = 128
  DIMS_PER_HEAD = 128
  PERCORE_BATCH_SIZE = 4
  MAX_SEQ_LEN = 1024
  VOCAB_SIZE = 32000
  FPROP_DTYPE = jnp.bfloat16
  USE_REPEATED_LAYER = True

  SUMMARY_INTERVAL_STEPS = 10
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  ICI_MESH_SHAPE = [1, 64, 8]

@experiment_registry.register
class C4Spmd512BAdam1024Replicas(c4.C4SpmdAdam):
  r"""GPT-3 config with 512B params. Model Parameters: 
  Global batch size = 1 * 64 * 16 * 2 = 2048"""
  NUM_LAYERS = 102
  MODEL_DIMS = 20480
  HIDDEN_DIMS = MODEL_DIMS * 4
  NUM_HEADS = 160
  DIMS_PER_HEAD = 128
  PERCORE_BATCH_SIZE = 2
  MAX_SEQ_LEN = 1024
  VOCAB_SIZE = 32000
  FPROP_DTYPE = jnp.bfloat16
  USE_REPEATED_LAYER = True

  SUMMARY_INTERVAL_STEPS = 10
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  ICI_MESH_SHAPE = [1, 64, 16]

@experiment_registry.register
class C4Spmd1024BAdam2048Replicas(c4.C4SpmdAdam):
  r"""GPT-3 config with 512B params. Model Parameters: 
  Global batch size = 1 * 256 * 8 * 1 = 4096"""
  NUM_LAYERS = 142
  MODEL_DIMS = 24576
  HIDDEN_DIMS = MODEL_DIMS * 4
  NUM_HEADS = 192
  DIMS_PER_HEAD = 128
  PERCORE_BATCH_SIZE = 2
  MAX_SEQ_LEN = 1024
  VOCAB_SIZE = 32000
  FPROP_DTYPE = jnp.bfloat16
  USE_REPEATED_LAYER = True

  SUMMARY_INTERVAL_STEPS = 10
  CHECKPOINT_POLICY = layers.AutodiffCheckpointType.SAVE_NOTHING
  ICI_MESH_SHAPE = [1, 256, 8]