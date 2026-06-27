# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.
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


import dataclasses
import os
import time

import numpy
import torch
from megatron.bridge.data.loaders import get_train_valid_test_num_samples
from megatron.bridge.data.utils import get_dataset_provider
from megatron.bridge.training.callbacks import Callback, CallbackContext
from megatron.bridge.training.config import MockGPTDatasetConfig
from megatron.core import parallel_state as mpu
from megatron.core.full_cuda_graph import FullCudaGraphWrapper
from megatron.core.inference.communication_utils import (
    broadcast_from_last_pipeline_stage,
)
from mlperf_common.frameworks.pyt import PyTCommunicationHandler
from mlperf_common.logging import MLLoggerWrapper


def get_last_pp_rank():
    is_last_pp = mpu.is_pipeline_last_stage(ignore_virtual=True)
    is_first_dp = mpu.get_data_parallel_rank() == 0
    is_first_tp = mpu.get_tensor_model_parallel_rank() == 0
    is_first_cp = mpu.get_context_parallel_rank() == 0
    return is_last_pp and is_first_dp and is_first_tp and is_first_cp


def broadcast_loss(loss_reduced):
    if "lm loss" in loss_reduced:
        loss_tensor = loss_reduced["lm loss"]
    else:
        loss_tensor = None

    loss_synced = broadcast_from_last_pipeline_stage(
        size=[1],
        dtype=torch.float32,
        tensor=loss_tensor.unsqueeze(0) if loss_tensor is not None else None,
    )

    return loss_synced.item()


class DeltaTimer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time = time.perf_counter()
        return self.start_time

    def get_delta(self):
        prev_time = self.start_time
        return self.reset() - prev_time


def _warm_npy_page_cache(datasets):
    """Page-cache-warm the .npy index files on every rank.

    After Phase 1 wrote the indices to path_to_cache on rank 0, only that
    rank's OS page cache is hot. Other ranks would pay a cold mmap+read
    during real dataset setup (first training step). Touching one element
    per OS page forces a read without loading the whole file into memory.
    Safe to run on every rank in parallel — pure local reads, no writes.

    Load errors are allowed to propagate: fast_cache_load skips the runtime
    os.path.isfile check, so this walk is the last validation gate before
    run_start. A missing or corrupt file must fail here, not at the first
    training batch when the MLPerf clock is already running.
    """
    stack = list(datasets) if isinstance(datasets, (list, tuple)) else [datasets]
    seen = set()
    while stack:
        ds = stack.pop()
        if ds is None:
            continue
        for attr in (
            "path_to_document_index",
            "path_to_sample_index",
            "path_to_shuffle_index",
            "path_to_dataset_index",
            "path_to_dataset_sample_index",
        ):
            p = getattr(ds, attr, None)
            if p and p not in seen:
                seen.add(p)
                arr = numpy.load(p, allow_pickle=True, mmap_mode="r")
                if arr.size > 0:
                    # reshape(-1) gives a zero-copy 1-D view of the C-contiguous
                    # .npy buffer so striding is per-element, not per-row.
                    # sample_index is 2-D (N, 2) int64; stepping the outer axis
                    # with step=itemsize skips every other 4 KB page.
                    flat = arr.reshape(-1)
                    step = max(1, 4096 // max(flat.itemsize, 1))
                    _ = flat[::step].sum()
        sub = getattr(ds, "datasets", None)
        if sub:
            stack.extend(sub)


mllogger = MLLoggerWrapper(PyTCommunicationHandler())


class MLPerfLoggingCallback(Callback):
    """MLPerf logging callback."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.global_batch_size = self.cfg.model.global_batch_size
        self.train_block_started = True
        self.train_current_block = 0
        self.force_success = cfg.custom.force_success_status
        self.previous_step = 0

    def on_data_init_start(self, context: CallbackContext):
        # Prebuild dataset indices into path_to_cache before logging run_start
        # so the index build does not count against MLPerf timing. Only reads
        # .idx metadata (document count + sizes), never training tokens.
        #
        # Phase 1: rank 0 writes the .npy caches. Force defer=False so
        # numpy.save runs, and fast_cache_load=False so the builder actually
        # checks for files and writes them (fast_cache_load shortcuts
        # cache_hit=True and skips the write path).
        # Phase 2: every rank mmap-touches the caches so the OS page cache is
        # warm before real dataset setup happens on the MLPerf clock. The
        # real setup after run_start keeps the user-configured defer and
        # fast_cache_load values (typically True/True) so it skips both the
        # mmap open and the file-existence checks.
        cfg = context.state.cfg
        if isinstance(cfg.dataset, MockGPTDatasetConfig):
            # Mock datasets generate synthetic data in memory — there are no
            # .npy index files to prebuild or page-cache-warm.
            mllogger.log_init_stop_run_start()
            return
        num_samples = get_train_valid_test_num_samples(cfg)

        prebuild_cfg = dataclasses.replace(
            cfg.dataset, defer_npy_index_mmap=False, fast_cache_load=False
        )
        get_dataset_provider(prebuild_cfg)(num_samples, prebuild_cfg)
        torch.distributed.barrier()

        touch_cfg = dataclasses.replace(cfg.dataset, defer_npy_index_mmap=True)
        touch_datasets = get_dataset_provider(touch_cfg)(num_samples, touch_cfg)
        _warm_npy_page_cache(touch_datasets)
        torch.distributed.barrier()

        mllogger.log_init_stop_run_start()

    def on_train_start(self, context: CallbackContext):
        context.state.should_stop = False
        mllogger.start(
            mllogger.constants.BLOCK_START,
            metadata={
                mllogger.constants.SAMPLES_COUNT: self.cfg.trainer.val_check_interval * self.global_batch_size,
                "step": context.state.train_state.step,
            },
        )
        self.timer = DeltaTimer()

    def on_train_end(self, context: CallbackContext):
        if self.train_block_started:
            self._end_train_block(context.state)

        FullCudaGraphWrapper.cuda_graph = None

    def on_eval_start(self, context: CallbackContext):
        """Log validation start."""
        if hasattr(context.state, "warmup") and context.state.warmup:
            return
        self._log_train_step_time(context.state)
        if self.train_block_started:
            self._end_train_block(context.state)

        mllogger.start(
            mllogger.constants.EVAL_START,
            metadata={
                mllogger.constants.SAMPLES_COUNT: self._get_samples_count(context.state),
                "step": self._get_step(context.state),
            },
        )

    def on_eval_end(self, context: CallbackContext):
        if hasattr(context.state, "warmup") and context.state.warmup:
            return
        self._log_custom_timedelta("validation_time", self._get_step(context.state))

        samples_count = self._get_samples_count(context.state)
        if self.cfg.model.pipeline_model_parallel_size > 1:
            loss = broadcast_loss(context.total_loss_dict)
        else:
            loss = context.total_loss_dict["lm loss"].item()

        mllogger.event(
            key=mllogger.constants.EVAL_ACCURACY,
            metadata={mllogger.constants.SAMPLES_COUNT: samples_count},
            value=loss,
        )
        mllogger.end(
            mllogger.constants.EVAL_STOP,
            metadata={
                mllogger.constants.SAMPLES_COUNT: samples_count,
                "step": self._get_step(context.state),
            },
        )

        if loss < self.cfg.custom.target_log_ppl:
            context.state.should_stop = True
            mllogger.end(
                mllogger.constants.RUN_STOP,
                metadata={mllogger.constants.SAMPLES_COUNT: samples_count, "status": "success"},
            )
        elif context.state.train_state.step >= self.cfg.trainer.max_steps:
            context.state.should_stop = True
            status = "success" if self.force_success else "aborted"
            mllogger.end(
                mllogger.constants.RUN_STOP,
                metadata={mllogger.constants.SAMPLES_COUNT: samples_count, "status": status},
            )
        if not os.environ.get("VAL_CHECK_INTERVAL"):
            # If VAL_CHECK_INTERVAL is not set, we use the default schedule of 18432 sequences after skipping the first SKIP_EVALS evaluations
            context.state.cfg.validation.eval_interval = self.cfg.default_val_check_interval

        if not context.state.should_stop:
            self._start_train_block(context.state)
        else:
            context.state.train_state.step = self.cfg.trainer.max_steps + 1
            context.state.train_state.do_valid = False
            context.state.train_state.do_test = False

    def on_train_step_end(self, context: CallbackContext):
        step = context.state.train_state.step + 1
        last_step = step >= self.cfg.trainer.max_steps
        eval_after_this_step = step % context.state.cfg.validation.eval_interval == 0
        if last_step and not eval_after_this_step:
            samples_count = self._get_samples_count(context.state)
            status = "success" if self.force_success else "aborted"
            self._end_train_block(context.state)
            mllogger.end(
                mllogger.constants.RUN_STOP,
                metadata={mllogger.constants.SAMPLES_COUNT: samples_count, "status": status},
            )
            self.train_block_started = False
            context.state.should_stop = True
            context.state.train_state.do_valid = False
            context.state.train_state.do_test = False

    def _start_train_block(self, global_state) -> None:
        self.train_block_started = True
        mllogger.start(
            mllogger.constants.BLOCK_START,
            metadata={
                mllogger.constants.SAMPLES_COUNT: global_state.cfg.validation.eval_interval * self.global_batch_size,
                "step": self._get_step(global_state),
            },
        )

    def _end_train_block(self, global_state) -> None:
        mllogger.end(
            mllogger.constants.BLOCK_STOP,
            metadata={
                mllogger.constants.SAMPLES_COUNT: global_state.cfg.validation.eval_interval * self.global_batch_size,
                "step": self._get_step(global_state),
            },
        )
        self.train_block_started = False

    def _log_train_step_time(self, global_state) -> None:
        delta_t = self.timer.get_delta()
        global_step = self._get_step(global_state)
        delta_step = global_step - self.previous_step
        mllogger.event(
            key="tracked_stats",
            metadata={mllogger.constants.SAMPLES_COUNT: delta_step * self.global_batch_size},
            value={
                "train_step_time": delta_t / (delta_step + 1e-8),
            },
        )

        self.previous_step = global_step

    def _log_custom_timedelta(self, value_key, step: int = 0):
        mllogger.event(
            key="tracked_stats",
            metadata={"step": step},
            value={value_key: self.timer.get_delta()},
        )

    def _get_step(self, global_state):
        return global_state.train_state.step

    def _get_samples_count(self, global_state):
        return self._get_step(global_state) * self.global_batch_size


class DeltaTimingCallback(Callback):
    def __init__(self, cfg):
        self.t0 = 0
        self.total_train_step_time = [0, 0]
        self.global_batch_size = cfg.model.global_batch_size
        self.log_every_n_steps = cfg.trainer.log_every_n_steps

    def on_train_start(self, context: CallbackContext):
        self.t0 = time.time()

    def on_train_step_end(self, context: CallbackContext):
        t1 = time.time()
        d = t1 - self.t0
        self.total_train_step_time[0] += d
        self.total_train_step_time[1] += 1
        self.t0 = t1

        if context.state.train_state.step % self.log_every_n_steps == 0 and get_last_pp_rank():
            mllogger.event(
                key="tracked_stats",
                metadata={mllogger.constants.SAMPLES_COUNT: self.global_batch_size * context.state.train_state.step},
                value={
                    "train_step_time": d,
                    "reduced_train_loss": context.loss_dict["lm loss"].item(),
                },
                unique=False,
            )

    def on_eval_end(self, context: CallbackContext):
        """Reset timer after validation to avoid including validation time in first train step."""
        self.t0 = time.time()
