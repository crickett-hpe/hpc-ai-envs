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

import os
from dataclasses import dataclass
from math import ceil
from typing import Callable

import hydra
import torch
import utils
from callback_logging import DeltaTimingCallback, MLPerfLoggingCallback, mllogger
from callback_warmup import WarmupCallback, create_mock_dataset_config
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.training.config import (
    CheckpointConfig,
    CommOverlapConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    DistributedInitConfig,
    GPTDatasetConfig,
    LoggerConfig,
    MixedPrecisionConfig,
    OptimizerConfig,
    ProfilingConfig,
    RerunStateMachineConfig,
    RNGConfig,
    SchedulerConfig,
    TrainingConfig,
    ValidationConfig,
)
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from omegaconf import OmegaConf


@dataclass
class Llama31ModelProvider(GPTModelProvider):
    normalization: str = "RMSNorm"
    activation_func: Callable = torch.nn.functional.silu
    gated_linear_unit: bool = True
    position_embedding_type: str = "rope"
    add_bias_linear: bool = False
    attention_dropout: float = 0.0
    hidden_dropout: float = 0.0
    share_embeddings_and_output_weights: bool = False
    bias_activation_fusion: bool = True
    masked_softmax_fusion: bool = True
    persist_layer_norm: bool = True
    bias_dropout_fusion: bool = True
    apply_rope_fusion: bool = True
    rotary_percent: float = 1.0
    rotary_base: int = 500_000
    rope_scaling: bool = True
    rope_scaling_factor: float = 8.0
    init_method_std: float = 0.01
    layernorm_epsilon: float = 1e-05
    num_query_groups: int = 8
    init_method_std: float = 0.02
    std: float = 0.02
    embedding_init_method_std: float = 0.02


@dataclass
class Llama31ModelProvider8B(Llama31ModelProvider):
    num_layers: int = 32
    hidden_size: int = 4096
    ffn_hidden_size: int = 14336
    num_attention_heads: int = 32


@dataclass
class Llama31ModelProvider405B(Llama31ModelProvider):
    num_layers: int = 126
    hidden_size: int = 16384
    ffn_hidden_size: int = 53248
    num_attention_heads: int = 128


OmegaConf.register_new_resolver("add", lambda x, y: x + y)
OmegaConf.register_new_resolver("multiply", lambda x, y: x * y)
OmegaConf.register_new_resolver("ceil_div", lambda x, y: (x + y - 1) // y)
OmegaConf.register_new_resolver("floor_div", lambda x, y: x // y)
OmegaConf.register_new_resolver("div", lambda x, y: x / y)
OmegaConf.register_new_resolver("if", lambda x, y, z: y if x else z)
OmegaConf.register_new_resolver("lt", lambda x, y: x < y)
OmegaConf.register_new_resolver("eq", lambda x, y: x == y)
OmegaConf.register_new_resolver("neq", lambda x, y: x != y)
OmegaConf.register_new_resolver("or", lambda *args: any(args))
OmegaConf.register_new_resolver("min", lambda x, y: min(x, y))
OmegaConf.register_new_resolver("floor", lambda x: int(x // 1))


def get_data(config):
    if config.model.data.mock_dataset:
        return create_mock_dataset_config(config)

    r = [6] if config.model.base_config == "8b" else [6, 7]
    train_datasets = [f"/preproc_data/c4-train.en_{idx}_text_document" for idx in r]
    train_datasets_weights = [50] * len(r)
    val_test_path = "/preproc_data/c4-validation-91205-samples.en_text_document"

    data_paths = [(train_datasets, train_datasets_weights), ([val_test_path], None), ([val_test_path], None)]

    return GPTDatasetConfig(
        blend_per_split=data_paths,
        sequence_length=config.model.encoder_seq_length,
        random_seed=config.model.seed,
        dataloader_type="single",
        num_workers=config.model.data.num_workers,
        path_to_cache="/npy_index",
        defer_npy_index_mmap=True,
        fast_cache_load=True,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        create_attention_mask=False,
    )


def get_model(config):
    base_config = config.model.base_config
    model_provider = Llama31ModelProvider8B if base_config == "8b" else Llama31ModelProvider405B

    tp = config.model.tensor_model_parallel_size
    ep = config.model.expert_model_parallel_size
    pp = config.model.pipeline_model_parallel_size
    pp_dtype = torch.bfloat16 if config.model.pipeline_model_parallel_size != 1 else None
    vp = config.model.virtual_pipeline_model_parallel_size
    cp = config.model.context_parallel_size
    sp = config.model.sequence_parallel
    # WAR for Megatron constraint: ETP * EP * PP must equal TP * CP * PP
    # for DP-last mapping compatibility.
    etp = tp * cp // ep
    asym_pp_embed = config.model.account_for_embedding_in_pipeline_split
    asym_pp_loss = config.model.account_for_loss_in_pipeline_split
    if config.model.overwritten_attributes.enable_cuda_graph:
        cuda_graph_impl = "local"
    else:
        cuda_graph_impl = "none"

    if config.model.overwritten_attributes.num_layers is not None:
        num_layers = config.model.overwritten_attributes.num_layers
        config.model.resume_from_checkpoint = None
    elif config.model.base_config == "405b":
        num_layers = 126
    else:
        num_layers = 32

    return model_provider(
        # Parallelism configuration
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        virtual_pipeline_model_parallel_size=vp,
        expert_tensor_parallel_size=etp,
        context_parallel_size=cp,
        sequence_parallel=sp,
        # Pipeline configuration
        pipeline_dtype=pp_dtype,
        account_for_embedding_in_pipeline_split=asym_pp_embed,
        account_for_loss_in_pipeline_split=asym_pp_loss,
        # Model configuration
        num_layers=num_layers,
        seq_length=config.model.encoder_seq_length,
        init_model_with_meta_device=config.model.mcore_fsdp,
        # Optimization features
        cross_entropy_loss_fusion=config.model.cross_entropy_loss_fusion,
        cross_entropy_fusion_impl=config.model.cross_entropy_fusion_impl,
        cuda_graph_impl=cuda_graph_impl,
        cuda_graph_scope=config.model.overwritten_attributes.cuda_graph_scope,
        fused_single_qkv_rope=config.model.fused_single_qkv_rope,
        gradient_accumulation_fusion=config.model.gradient_accumulation_fusion,
        tp_only_amax_red=config.model.tp_only_amax_red,
        use_te_rng_tracker=config.model.use_te_rng_tracker,
        use_transformer_engine_op_fuser=config.model.use_transformer_engine_op_fuser,
        cuda_graph_warmup_steps=config.model.custom.cuda_graph_warmup_steps,
        # CPU offloading
        cpu_offloading=config.model.cpu_offloading,
        cpu_offloading_num_layers=config.model.cpu_offloading_num_layers,
        cpu_offloading_weights=False,
    )


def get_mixed_precision(config):
    # fp8 knobs:
    fp8_type = None
    if config.model.fp8:
        fp8_type = "hybrid" if config.model.fp8_hybrid else "e4m3"
    fp8_margin = 0
    fp8_amax_history_len = config.model.fp8_amax_history_len
    fp8_amax_compute_algo = config.model.fp8_amax_compute_algo
    fp8_param_gather = config.model.optim.fp8_param_gather
    fp8_recipe = config.model.fp8_recipe
    # fp4 knobs:
    fp4_type = None
    if config.model.fp4:
        fp4_type = "e2m1"
    fp4_recipe = config.model.fp4_recipe

    first_last_layers_bf16 = config.model.first_last_layers_bf16
    num_layers_at_start_in_bf16 = config.model.num_layers_at_start_in_bf16
    num_layers_at_end_in_bf16 = config.model.num_layers_at_end_in_bf16

    if config.model.fp8 or config.model.fp4:
        mixed_precision = MixedPrecisionConfig(
            bf16=True,
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            autocast_enabled=False,
            grad_reduce_in_fp32=False,
            # fp8
            fp8=fp8_type,
            fp8_margin=fp8_margin,
            fp8_recipe=fp8_recipe,
            fp8_amax_history_len=fp8_amax_history_len,
            fp8_amax_compute_algo=fp8_amax_compute_algo,
            fp8_param_gather=fp8_param_gather,
            fp8_dot_product_attention=config.model.fp8_dot_product_attention,
            # fp4
            fp4=fp4_type,
            fp4_recipe=fp4_recipe,
            # First/last layers in bf16
            first_last_layers_bf16=first_last_layers_bf16,
            num_layers_at_start_in_bf16=num_layers_at_start_in_bf16,
            num_layers_at_end_in_bf16=num_layers_at_end_in_bf16,
        )
    else:
        mixed_precision = MixedPrecisionConfig(
            precision="bf16-mixed",
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            autocast_enabled=False,
            grad_reduce_in_fp32=False,
        )

    return mixed_precision


def get_ddp_config(config):
    overlap_grad_reduce = config.model.optim.overlap_grad_reduce
    overlap_param_gather = config.model.optim.overlap_param_gather
    align_param_gather = config.model.optim.align_param_gather
    use_distributed_optimizer = config.model.optim.use_distributed_optimizer
    bucket_size = config.model.optim.bucket_size
    num_distributed_optimizer_instances = config.model.optim.num_distributed_optimizer_instances
    fp8_param_gather = config.model.optim.fp8_param_gather
    nccl_ub_dp = config.model.optim.nccl_ub_dp
    outer_dp_sharding_strategy = config.model.optim.outer_dp_sharding_strategy
    fsdp_manual_registration = config.model.optim.fsdp_manual_registration

    return DistributedDataParallelConfig(
        check_for_nan_in_grad=False,
        grad_reduce_in_fp32=False,
        average_in_collective=False,
        bucket_size=bucket_size,
        # Overlap
        overlap_grad_reduce=overlap_grad_reduce,
        overlap_param_gather=overlap_param_gather,
        align_param_gather=align_param_gather,
        # Distributed optimizer
        use_distributed_optimizer=use_distributed_optimizer,
        num_distributed_optimizer_instances=num_distributed_optimizer_instances,
        data_parallel_sharding_strategy="optim_grads_params",
        outer_dp_sharding_strategy=outer_dp_sharding_strategy,
        # FP8
        fp8_param_gather=fp8_param_gather,
        keep_fp8_transpose_cache=False,
        # FSDP
        fsdp_double_buffer=nccl_ub_dp,
        use_megatron_fsdp=config.model.mcore_fsdp,
        # NCCL
        nccl_ub=nccl_ub_dp,
        fsdp_manual_registration=fsdp_manual_registration,
    )


def get_dist_config(config):
    nccl_communicator_config_path = config.model.nccl_communicator_config_path

    # Using FSDP in MBridge requires Gloo process groups.
    # https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/4f21a31d7c859548d62dc1b0a63a325349ce1a93/src/megatron/bridge/training/config.py#L1274-L1280
    use_gloo_process_groups = config.model.mcore_fsdp

    return DistributedInitConfig(
        nccl_communicator_config_path=nccl_communicator_config_path,
        use_tp_pp_dp_mapping=config.model.use_tp_pp_dp_mapping,
        use_sharp=config.model.sharp,
        use_gloo_process_groups=use_gloo_process_groups,
    )


def get_optimizer_config(config):
    return OptimizerConfig(
        optimizer="adam",
        lr=config.model.optim.lr,
        weight_decay=0.1,
        bf16=config.trainer.precision == "bf16",
        fp16=config.trainer.precision == "fp16",
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-5,
        use_distributed_optimizer=True,
        clip_grad=1.0,
        min_lr=config.model.optim.sched.min_lr,
    )


def get_scheduler_config(config):
    if config.model.base_config == "405b":
        decay_steps = ceil(
            int(os.environ.get("OPT_LR_DECAY_STEPS", 0)) * 1152.0 / config.model.global_batch_size
        ) - int(config.model.optim.sched.warmup_steps)
    else:
        decay_steps = int(os.environ.get("OPT_LR_DECAY_STEPS", 0)) - int(config.model.optim.sched.warmup_steps)

    return SchedulerConfig(
        lr_decay_style="cosine",
        lr_decay_iters=decay_steps,
        lr_warmup_iters=config.model.optim.sched.warmup_steps,
        start_weight_decay=0.1,
        end_weight_decay=0.1,
    )


def get_train_config(config):
    return TrainingConfig(
        global_batch_size=config.model.global_batch_size,
        micro_batch_size=config.model.micro_batch_size,
        rampup_batch_size=None,
        train_iters=config.trainer.max_steps,
    )


def get_validation_config(config):
    return ValidationConfig(
        eval_interval=config.trainer.val_check_interval,
        eval_iters=config.trainer.eval_iters,
    )


def get_tokenizer_config(config):
    if config.model.data.mock_dataset:
        vocab_size = config.model.data.mock_tokenizer_vocab_size
        return TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=vocab_size)

    return TokenizerConfig(
        tokenizer_type="HuggingFaceTokenizer",
        tokenizer_model=config.model.tokenizer.model,
        hf_tokenizer_kwargs={"use_fast": True},
    )


def get_checkpoint_config(config):
    ckpt_path = config.model.resume_from_checkpoint if config.model.base_config == "405b" else None
    return CheckpointConfig(
        load=ckpt_path,
        save=None,
        fully_parallel_load=config.model.dist_ckpt_parallel_load,
        dist_ckpt_strictness="log_all",
        load_optim=False,
        load_rng=False,
        load_main_params_from_ckpt=config.model.optim.fp8_param_gather,
        ckpt_format=config.model.dist_ckpt_format,
    )


def get_rerun_state_machine_config(config):
    return RerunStateMachineConfig(
        rerun_mode="disabled",
        check_for_nan_in_loss=False,
        check_for_spiky_loss=False,
    )


def get_logger_config(config):
    return LoggerConfig(log_interval=config.trainer.max_steps + 1)


def get_rng_config(config):
    return RNGConfig(seed=config.model.seed, te_rng_tracker=config.model.use_te_rng_tracker)


def get_comm_overlap_config(config):
    tp_comm_overlap = config.model.ub_tp_comm_overlap

    tp_comm_overlap_cfg = None
    if tp_comm_overlap:
        from megatron.bridge.training.comm_overlap import (
            BulkOverlapCfg,
            PipelineOverlapCfg,
            RingExchangeOverlapCfg,
            TransformerLayerTPOverlapCfg,
        )

        buffer_options = config.model.ub_tp_comm_overlap_cfg
        userbuffer_args = {}

        for key in [
            "qkv_dgrad",
            "qkv_wgrad",
            "fc1_dgrad",
            "fc1_wgrad",
            "qkv_fprop",
            "proj_dgrad",
            "fc1_fprop",
            "fc2_dgrad",
            "proj_fprop",
            "fc2_fprop",
        ]:
            if key in buffer_options:
                attributes = buffer_options[key]
                fp8_buf = False
                try:
                    fp8_buf = bool(attributes.fp8_buf)
                except:
                    pass
                if attributes.method == "pipeline":
                    userbuffer_args[key] = PipelineOverlapCfg(
                        num_sm=attributes.num_sm,
                        cga_size=attributes.cga_size,
                        num_splits=attributes.num_splits,
                        set_sm_margin=bool(attributes.set_sm_margin),
                        fp8_buf=fp8_buf,
                    )
                elif attributes.method == "bulk":
                    userbuffer_args[key] = BulkOverlapCfg(
                        num_sm=attributes.num_sm,
                        cga_size=attributes.cga_size,
                        set_sm_margin=bool(attributes.set_sm_margin),
                    )
                elif attributes.method == "ring_exchange":
                    userbuffer_args[key] = RingExchangeOverlapCfg(
                        fp8_buf=fp8_buf,
                    )
                else:
                    assert False, f"method {attributes.method} is not defined."

        tp_comm_overlap_cfg = TransformerLayerTPOverlapCfg(**userbuffer_args)

    return CommOverlapConfig(
        tp_comm_overlap=tp_comm_overlap,
        tp_comm_overlap_cfg=tp_comm_overlap_cfg,
        # PP overlap
        overlap_p2p_comm=config.model.overlap_p2p_comm,
        batch_p2p_comm=config.model.batch_p2p_comm,
        # DP overlap
        overlap_grad_reduce=config.model.optim.overlap_grad_reduce,
        overlap_param_gather=config.model.optim.overlap_param_gather,
        overlap_param_gather_with_optimizer_step=config.model.optim.overlap_param_gather_with_optim_step,
        align_param_gather=config.model.optim.align_param_gather,
        bucket_size=config.model.optim.bucket_size,
        # Pipeline bubble
        defer_embedding_wgrad_compute=config.model.defer_embedding_wgrad_compute,
        wgrad_deferral_limit=config.model.wgrad_deferral_limit,
    )


def get_profiling_config(config):
    if config.misc.memory_profiler.enable:
        return ProfilingConfig(
            use_pytorch_profiler=True,
            record_memory_history=True,
            memory_snapshot_path=f"{config.misc.memory_profiler.file_prefix}.pickle",
            profile_step_start=config.misc.memory_profiler.start_step,
            profile_step_end=config.misc.memory_profiler.end_step,
        )
    return ProfilingConfig(
        use_nsys_profiler=config.model.nsys_profile.enabled,
        profile_step_start=config.model.nsys_profile.start_step,
        profile_step_end=config.model.nsys_profile.end_step,
        profile_ranks=[int(r) for r in str(config.model.nsys_profile.ranks).split(",")],
        record_shapes=config.model.nsys_profile.gen_shape,
        nvtx_ranges=config.model.nsys_profile.nvtx_ranges,
    )


def create_config(config):
    return ConfigContainer(
        checkpoint=get_checkpoint_config(config),
        comm_overlap=get_comm_overlap_config(config),
        dataset=get_data(config),
        ddp=get_ddp_config(config),
        dist=get_dist_config(config),
        logger=get_logger_config(config),
        mixed_precision=get_mixed_precision(config),
        model=get_model(config),
        optimizer=get_optimizer_config(config),
        rng=get_rng_config(config),
        profiling=get_profiling_config(config),
        rerun_state_machine=get_rerun_state_machine_config(config),
        scheduler=get_scheduler_config(config),
        tokenizer=get_tokenizer_config(config),
        train=get_train_config(config),
        validation=get_validation_config(config),
    )


def log_hyperparams(config):
    if config.model.base_config == "405b":
        bmark = mllogger.constants.LLAMA31_405B
        opt_lr_decay_steps = ceil(
            int(os.environ.get("OPT_LR_DECAY_STEPS", 0)) * 1152.0 / config.model.global_batch_size
        ) - int(config.model.optim.sched.warmup_steps)
    else:
        bmark = mllogger.constants.LLAMA31_8B
        opt_lr_decay_steps = int(os.environ.get("OPT_LR_DECAY_STEPS", 0)) - int(config.model.optim.sched.warmup_steps)
    mllogger.mlperf_submission_log(bmark)

    # Collects configs to be logged
    logging_configs = {
        # seeds
        mllogger.constants.SEED: config.model.seed,
        # HPs
        mllogger.constants.GLOBAL_BATCH_SIZE: config.model.global_batch_size,
        mllogger.constants.GRADIENT_ACCUMULATION_STEPS: (int(os.environ["MINIBS"]) / config.model.micro_batch_size),
        mllogger.constants.MAX_SEQUENCE_LENGTH: config.model.encoder_seq_length,
        mllogger.constants.EVAL_SAMPLES: int(os.environ.get("VAL_SAMPLES", 0)),
        mllogger.constants.TRAIN_SAMPLES: 1574207408,
        mllogger.constants.INIT_CHECKPOINT_STEP: config.model.custom.init_global_step,
        # Optimizers
        mllogger.constants.OPT_NAME: mllogger.constants.ADAMW,
        mllogger.constants.OPT_BASE_LR: config.model.optim.lr,
        mllogger.constants.OPT_ADAMW_BETA_1: 0.9,
        mllogger.constants.OPT_ADAMW_BETA_2: 0.95,
        mllogger.constants.OPT_ADAMW_EPSILON: 1e-5,
        mllogger.constants.OPT_ADAMW_WEIGHT_DECAY: 0.1,
        mllogger.constants.OPT_GRADIENT_CLIP_NORM: 1.0,
        # Schedulers
        mllogger.constants.OPT_END_LR: config.model.optim.sched.min_lr,
        mllogger.constants.OPT_LR_WARMUP_STEPS: config.model.optim.sched.warmup_steps,
        mllogger.constants.OPT_LR_DECAY_STEPS: opt_lr_decay_steps,
        mllogger.constants.MAX_STEPS: int(os.environ.get("MAX_STEPS", 0)),
        mllogger.constants.OPT_LR_DECAY_SCHEDULE: "cosine with linear warmup",
        # custom
        "target_accuracy": config.custom.target_log_ppl,
    }

    for key, value in logging_configs.items():
        mllogger.event(key=key, value=value)


@hydra.main(config_path="conf", config_name="llama31_config_custom", version_base="1.2")
def main(cfg):
    OmegaConf.resolve(cfg)
    config_container = create_config(cfg)
    if utils.rank == 0:
        log_hyperparams(cfg)
    callbacks = [
        WarmupCallback(cfg, forward_step_func=forward_step),
        DeltaTimingCallback(cfg),
        MLPerfLoggingCallback(cfg),
    ]

    # GC Config
    config_container.train.manual_gc = True
    config_container.train.manual_gc_interval = 500
    config_container.train.manual_gc_eval = False

    # Memory management
    config_container.train.empty_unused_memory_level = 0
    config_container.train.train_sync_interval = None

    # Skip numeric checks
    config_container.train.check_optimizer_step_success = False
    config_container.train.skip_sync_grad_norm_across_mp = True
    
    # Skip logging & timers
    config_container.logger.skip_train_metrics_log = True
    config_container.logger.timing_log_level = -1

    config_container.train.exit_signal_handler = False

    pretrain(config_container, forward_step_func=forward_step, callbacks=callbacks)


if __name__ == "__main__":
    if utils.rank == 0:
        mllogger.start(key=mllogger.constants.INIT_START)
    main()
