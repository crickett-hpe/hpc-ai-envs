# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import logging
from math import floor, ceil


def get_rank():
    return int(os.getenv("SLURM_PROCID", 0))


class RankZeroFilter(logging.Filter):
    def filter(self, record):
        return get_rank() == 0


root = logging.getLogger()
root.addFilter(RankZeroFilter())


import warnings

import hydra
import torch
from pytorch_lightning.utilities import rank_zero_only

torch.cuda.set_device(int(os.getenv("SLURM_LOCALID", "0")))

import nemo.lightning as nl
from custom_callbacks import (
    DeltaTimingCallback,
    MetricsLogger,
    PrintArtifacts,
    setup_auxiliary_loggers,
    MemoryProfileCallback,
)
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from mlperf_common.callbacks import mllogger
from mocking import MockDataModuleWithBatchLimit, MockTokenizer
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.llm.api import train
from nemo.collections.llm.gpt.data import PreTrainingDataModule
from nemo.collections.llm.gpt.model import LlamaModel
from nemo.collections.llm.gpt.model.llama import (
    Llama31Config8B,
    Llama31Config70B,
    Llama31Config405B,
)
from nemo.lightning.pytorch.callbacks import GarbageCollectionCallback, NsysCallback
from nemo.lightning.pytorch.optim import (
    CosineAnnealingScheduler,
    MegatronOptimizerModule,
)
from nemo.lightning.pytorch.plugins.mixed_precision import MegatronMixedPrecision
from nemo.utils import logging
from omegaconf.omegaconf import OmegaConf

OmegaConf.register_new_resolver("add", lambda x, y: x + y)
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


# ================ Components ================
# ======== Data ========


def mock_data(config, for_warmup=False):
    # Parameters
    gbs = config.model.global_batch_size
    mbs = config.model.micro_batch_size
    seq_length = config.model.encoder_seq_length
    tokenizer_path = config.model.tokenizer.model

    tokenizer = None
    if tokenizer_path != "":
        tokenizer = AutoTokenizer(pretrained_model_name=tokenizer_path)
    else:
        tokenizer = MockTokenizer(
            vocab_size=config.model.data.mock_tokenizer_vocab_size
        )
    if for_warmup:
        return MockDataModuleWithBatchLimit(
            global_batch_size=gbs,
            micro_batch_size=mbs,
            tokenizer=tokenizer,
            seq_length=seq_length,
            num_workers=8,
        )

    num_train_samples = int(config.trainer.max_steps * gbs)
    eval_iters = (
        config.trainer.max_steps // config.trainer.val_check_interval + 1
    ) * config.trainer.limit_val_batches
    num_val_samples = int(eval_iters * gbs)
    num_test_samples = int(config.trainer.limit_test_batches * gbs)

    return MockDataModuleWithBatchLimit(
        global_batch_size=gbs,
        micro_batch_size=mbs,
        tokenizer=tokenizer,
        seq_length=seq_length,
        num_workers=8,
        num_train_samples=num_train_samples,
        num_val_samples=num_val_samples,
        num_test_samples=num_test_samples,
    )


def get_data(config):
    # Parameters
    gbs = config.model.global_batch_size
    mbs = config.model.micro_batch_size
    seq_length = config.model.encoder_seq_length
    tokenizer_path = config.model.tokenizer.model
    seed = config.model.seed

    tokenizer = AutoTokenizer(pretrained_model_name=tokenizer_path)

    val_test_path = "/preproc_data/c4-validation-91205-samples.en_text_document"

    if config.model.base_config == "8b":
        r = [6]
    elif config.model.base_config == "70b":
        r = [6, 7]
    elif config.model.base_config == "405b":
        r = [6, 7]
    train_datasets = sum(
        [
            ["50", f"/preproc_data/c4-train.en_{idx}_text_document"]
            for idx in r
        ],
        [],
    )
    data_paths = {
        "train": train_datasets,
        "validation": [
            val_test_path,
        ],
        "test": [
            val_test_path,
        ],
    }

    return PreTrainingDataModule(
        tokenizer=tokenizer,
        paths=data_paths,
        num_workers=8,  # TODO: make it configurable
        seq_length=seq_length,
        global_batch_size=gbs,
        micro_batch_size=mbs,
        index_mapping_dir="/npy_index",
        seed=seed,
        # Option to reset the position IDs in the dataset at an interval.
        reset_position_ids=False,
        # Option to reset the attention mask from the dataset.
        reset_attention_mask=False,
        # Option to enable the EOD mask loss.
        eod_mask_loss=False,
        # Rampup batch size, should be in format of [start_global_batch_size, batch_size_increment, ramup_samples].
        rampup_batch_size=None,
    )


# ======== Model ========


def get_model_with_precision(config, tokenizer):
    base_config = config.model.base_config
    seq_length = config.model.encoder_seq_length
    customized_config = config.model.overwritten_attributes

    # overwrites VP
    overwritten_vp = config.model.virtual_pipeline_model_parallel_size

    assert not (config.model.fp8 and config.model.fp4), "fp8 and fp4 cannot be enabled at the same time"
    # fp8 knobs:
    fp8_type = None
    if config.model.fp8:
        fp8_type = "hybrid" if config.model.fp8_hybrid else "e4m3"
    fp8_margin = 0
    fp8_amax_history_len = config.model.fp8_amax_history_len
    fp8_amax_compute_algo = config.model.fp8_amax_compute_algo
    tp_only_amax_red = config.model.tp_only_amax_red
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

    # Model part
    base_llama_config = None
    if base_config == "8b":
        base_llama_config = Llama31Config8B(
            seq_length=seq_length,
            gradient_accumulation_fusion=config.model.gradient_accumulation_fusion,
        )
    elif base_config == "70b":
        base_llama_config = Llama31Config70B(seq_length=seq_length)
    elif base_config == "405b":
        base_llama_config = Llama31Config405B(
            fp8=fp8_type,
            fp8_recipe=fp8_recipe,  # Must be set during init so Fiddle captures it correctly
            seq_length=seq_length,
            num_layers=(
                customized_config.num_layers
                if customized_config.num_layers is not None
                else 126
            ),
            gradient_accumulation_fusion=config.model.gradient_accumulation_fusion,
        )
    else:
        assert False, "Unsupported base config type: " + base_config

    if overwritten_vp is not None:
        base_llama_config.virtual_pipeline_model_parallel_size = overwritten_vp

    if fp8_type is not None:
        base_llama_config.fp8 = fp8_type
        base_llama_config.fp8_margin = fp8_margin
        base_llama_config.fp8_amax_history_len = fp8_amax_history_len
        base_llama_config.fp8_amax_compute_algo = fp8_amax_compute_algo
        base_llama_config.tp_only_amax_red = tp_only_amax_red

    base_llama_config.enable_cuda_graph = customized_config.enable_cuda_graph
    base_llama_config.cuda_graph_scope = customized_config.cuda_graph_scope
    base_llama_config.use_transformer_engine_op_fuser = config.model.use_transformer_engine_op_fuser
    base_llama_config.cross_entropy_fusion_impl = config.model.cross_entropy_fusion_impl
    base_llama_config.fused_single_qkv_rope = config.model.fused_single_qkv_rope
    base_llama_config.fp8_dot_product_attention = config.model.fp8_dot_product_attention
    model = LlamaModel(config=base_llama_config, tokenizer=tokenizer)
    model.cross_entropy_loss_fusion = config.model.cross_entropy_loss_fusion

    precision = None
    # Precision part
    if config.model.fp8 or config.model.fp4:  # FP8 or FP4 takes precedences
        precision = MegatronMixedPrecision(
            precision="bf16-mixed",
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
            # fp4
            #fp4=fp4_type,
            #fp4_recipe=fp4_recipe,
            first_last_layers_bf16=first_last_layers_bf16,
            num_layers_at_start_in_bf16=num_layers_at_start_in_bf16,
            num_layers_at_end_in_bf16=num_layers_at_end_in_bf16,
            fp8_dot_product_attention = config.model.fp8_dot_product_attention
        )
    elif config.trainer.precision == "bf16":
        precision = MegatronMixedPrecision(
            precision="bf16-mixed",
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            autocast_enabled=False,
            grad_reduce_in_fp32=False,
        )
    else:
        assert False, f"Unsupported precision {config.trainer.precision}"

    return model, precision


def get_optimizer(config):
    # Optimizer params
    lr = config.model.optim.lr
    bf16 = config.trainer.precision == "bf16"
    fp16 = config.trainer.precision == "fp16"

    # Scheduler params
    warmup_steps = config.model.optim.sched.warmup_steps
    min_lr = config.model.optim.sched.min_lr

    optimizer_config = OptimizerConfig(
        optimizer="adam",
        lr=lr,
        # TODO: make all of them configurable?
        weight_decay=0.1,
        bf16=bf16,
        fp16=fp16,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-5,
        use_distributed_optimizer=True,
        clip_grad=1.0,
    )

    sched = CosineAnnealingScheduler(
        warmup_steps=warmup_steps,
        constant_steps=0,
        min_lr=min_lr,
    )

    return MegatronOptimizerModule(config=optimizer_config, lr_scheduler=sched)


# ======== Trainer related ========


def get_strategy(config):
    from megatron.core.dist_checkpointing.validation import StrictHandling

    tp = config.model.tensor_model_parallel_size
    ep = config.model.expert_model_parallel_size
    pp = config.model.pipeline_model_parallel_size
    pp_dtype = (
        torch.bfloat16 if config.model.pipeline_model_parallel_size != 1 else None
    )
    vp = config.model.virtual_pipeline_model_parallel_size
    cp = config.model.context_parallel_size
    sp = config.model.sequence_parallel
    # WAR for Megatron constraint: ETP * EP * PP must equal TP * CP * PP 
    # for DP-last mapping compatibility.
    etp = tp * cp // ep

    asym_pp_embed = config.model.account_for_embedding_in_pipeline_split
    asym_pp_loss = config.model.account_for_loss_in_pipeline_split

    use_tp_pp_dp_mapping = config.model.use_tp_pp_dp_mapping
    ckpt_dist_load = config.model.dist_ckpt_parallel_load

    overlap_grad_reduce = config.model.optim.overlap_grad_reduce
    overlap_param_gather = config.model.optim.overlap_param_gather
    align_param_gather = config.model.optim.align_param_gather
    use_distributed_optimizer = config.model.optim.use_distributed_optimizer
    bucket_size = config.model.optim.bucket_size
    fp8_param_gather = config.model.optim.fp8_param_gather
    use_te_rng_tracker = (config.model.use_te_rng_tracker,)
    nccl_communicator_config_path=config.model.nccl_communicator_config_path
    
    # PLT moves model to CPU at the end of training which takes some time
    # This patch removes that move
    def teardown_patch(self):
        return
    nl.MegatronStrategy.teardown = teardown_patch
    
    return nl.MegatronStrategy(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        pipeline_dtype=pp_dtype,
        virtual_pipeline_model_parallel_size=vp,
        expert_tensor_parallel_size=etp,
        context_parallel_size=cp,
        use_te_rng_tracker=use_te_rng_tracker,
        sequence_parallel=sp,
        use_tp_pp_dp_mapping=use_tp_pp_dp_mapping,
        nccl_communicator_config_path=nccl_communicator_config_path,
        account_for_embedding_in_pipeline_split=asym_pp_embed,
        account_for_loss_in_pipeline_split=asym_pp_loss,
        gradient_as_bucket_view=True,
        ckpt_load_optimizer=not fp8_param_gather, # handle case when the precision of checkpoints is higher than model params
        ckpt_load_main_params=fp8_param_gather, # handle case when the precision of checkpoints is higher than model params
        ckpt_async_save=True,
        ckpt_load_strictness=StrictHandling.LOG_ALL,
        ckpt_parallel_load=ckpt_dist_load,
        ddp=DistributedDataParallelConfig(
            check_for_nan_in_grad=False,
            grad_reduce_in_fp32=False,
            overlap_grad_reduce=overlap_grad_reduce,
            overlap_param_gather=overlap_param_gather,
            align_param_gather=align_param_gather,
            use_distributed_optimizer=use_distributed_optimizer,
            bucket_size=bucket_size,
            average_in_collective=True,
            fp8_param_gather=fp8_param_gather,
        ),
    )


def get_trainer(config, precision_plugin, strategy, warmup_config_overwrite=False):
    num_nodes = config.trainer.num_nodes
    devices = config.trainer.devices

    limit_train_batches = config.trainer.limit_train_batches
    limit_test_batches = config.trainer.limit_test_batches
    limit_val_batches = config.trainer.limit_val_batches
    max_steps = config.trainer.max_steps

    log_every_n_steps = config.trainer.log_every_n_steps
    val_check_interval = config.trainer.val_check_interval
    num_sanity_val_steps = config.trainer.num_sanity_val_steps

    enable_progress_bar = config.trainer.enable_progress_bar

    if warmup_config_overwrite:
        # max_steps = config.model.custom.warmup_train_steps
        limit_train_batches = 192
        limit_test_batches = 1
        limit_val_batches = 0
        log_every_n_steps = 1
        max_steps = 5
        val_check_interval = 100
        num_sanity_val_steps = 5

    return nl.Trainer(
        accelerator="gpu",
        num_nodes=num_nodes,
        devices=devices,
        callbacks=[],
        accumulate_grad_batches=1,
        limit_train_batches=limit_train_batches,
        limit_test_batches=limit_test_batches,
        limit_val_batches=limit_val_batches,
        log_every_n_steps=log_every_n_steps,
        val_check_interval=val_check_interval,
        num_sanity_val_steps=num_sanity_val_steps,
        max_steps=max_steps,
        plugins=precision_plugin,
        strategy=strategy,
        use_distributed_sampler=False,
        enable_checkpointing=False,
        logger=False,
        benchmark=False,
        enable_model_summary=True,
        enable_progress_bar=enable_progress_bar,
    )


def get_logger(config):
    save_last = config.exp_manager.checkpoint_callback_params.save_last
    save_top_k = config.exp_manager.save_top_k
    every_n_epochs = config.exp_manager.every_n_epochs

    name = config.model.name
    log_dir = config.exp_manager.explicit_log_dir

    ckpt = nl.ModelCheckpoint(
        save_last=save_last,
        save_top_k=save_top_k,
        every_n_epochs=every_n_epochs,
    )

    return nl.NeMoLogger(
        ckpt=ckpt, name=name, tensorboard=None, wandb=None, log_dir=log_dir
    )


def get_overlap_callback(config):
    ub_tp_comm_overlap = config.model.ub_tp_comm_overlap

    # Performance knobs
    userbuffer_config = None
    if ub_tp_comm_overlap:
        from nemo.collections.llm.recipes.tp_overlap_configs.userbuffers import (
            BulkOverlapCfg,
            PipelineOverlapCfg,
            RingExchangeOverlapCfg,
            TransformerLayerTPOverlapCfg,
        )

        buffer_options = config.model.ub_tp_comm_overlap_cfg
        # keys are qkv_dgrad, qkv_wgrad, fc1_dgrad, fc1_wgrad, qkv_fprop, proj_dgrad, fc1_fprop, fc2_dgrad, proj_fprop, fc2_fprop
        userbuffer_args = {}
        expected_args_list = [
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
        ]
        assert all([x in buffer_options for x in expected_args_list]), (
            f"{', '.join([x for x in buffer_options if x not in expected_args_list])} are not in expected values list"
        )
        for key in buffer_options.keys():
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

        userbuffer_config = TransformerLayerTPOverlapCfg(**userbuffer_args)

    from nemo.lightning.pytorch.callbacks.megatron_comm_overlap import (
        MegatronCommOverlapCallback,
    )

    comm_overlap = MegatronCommOverlapCallback(
        tp_comm_overlap=config.model.ub_tp_comm_overlap,
        tp_comm_overlap_cfg=userbuffer_config,
        overlap_p2p_comm=config.model.overlap_p2p_comm,
        batch_p2p_comm=config.model.batch_p2p_comm,
        overlap_grad_reduce=config.model.optim.overlap_grad_reduce,
        overlap_param_gather=config.model.optim.overlap_param_gather,
        overlap_param_gather_with_optimizer_step=config.model.optim.overlap_param_gather_with_optim_step,
        align_param_gather=config.model.optim.align_param_gather,
        defer_embedding_wgrad_compute=config.model.defer_embedding_wgrad_compute,
        wgrad_deferral_limit=config.model.wgrad_deferral_limit,
        bucket_size=config.model.optim.bucket_size,
    )

    return comm_overlap


def get_autoresume(config):
    return nl.AutoResume(
        restore_config=nl.RestoreConfig(path=config.model.resume_from_checkpoint)
    )


@rank_zero_only
def log_hyperparams(config):
    if config.model.base_config == "405b":
        bmark = mllogger.constants.LLAMA31_405B
        opt_lr_decay_steps = ceil(int(os.environ.get("OPT_LR_DECAY_STEPS", 0)) * 1152.0/config.model.global_batch_size) - int(config.model.optim.sched.warmup_steps)
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
        mllogger.constants.GRADIENT_ACCUMULATION_STEPS: (
            int(os.environ["MINIBS"]) / config.model.micro_batch_size
        ),
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
        "target_accuracy": config.model.custom.target_log_ppl
    }

    for key, value in logging_configs.items():
        mllogger.event(key=key, value=value)


# ================ Training ================


def set_breakpoint_rank0():
    from nemo.utils.get_rank import is_global_rank_zero

    if is_global_rank_zero():
        import pdb

        pdb.set_trace()


@hydra.main(config_path="conf", config_name="llama31_config_custom", version_base="1.2")
def main(cfg):
    # Suppress warnings
    warnings.filterwarnings("ignore")
    torch.set_warn_always(False)

    # Read MLPerf Configuration
    OmegaConf.resolve(cfg)

    import logging as base_logging

    base_logging.getLogger("torch.distributed.distributed_c10d").setLevel(
        logging.WARNING
    )
    base_logging.getLogger("DotProductAttention").setLevel(base_logging.WARNING)

    if cfg.model.nsys_profile.enabled and os.getenv("PROFILE_RANKS", "") != "":
        prof_ranks = [
            int(rank) for rank in os.getenv("PROFILE_RANKS").replace(" ", "").split(",")
        ]
        cfg.model.nsys_profile.ranks = prof_ranks

    logging.info("\n\n**************** Experiment configuration ****************")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    logging.info(
        f"\nTP: {cfg.model.tensor_model_parallel_size}; PP: {cfg.model.pipeline_model_parallel_size}; VP: {cfg.model.virtual_pipeline_model_parallel_size}; CP: {cfg.model.context_parallel_size}"
    )

    # Components
    logging.info("======== Benchmarked setups ========")

    # Pre-training warmup components
    warmup_data = mock_data(cfg, for_warmup=True)

    # Actual training components
    data = None
    if cfg.model.data.mock_dataset:
        data = mock_data(cfg, for_warmup=False)
    else:
        data = get_data(cfg)
    model, precision = get_model_with_precision(cfg, data.tokenizer)
    optimizer = get_optimizer(cfg)
    strategy = get_strategy(cfg)
    trainer = get_trainer(cfg, precision, strategy)
    logger = None if cfg.model.custom.disable_nemo_logs else get_logger(cfg)
    resume = None if cfg.model.resume_from_checkpoint is None else get_autoresume(cfg)

    # Callbacks
    trainer.callbacks.append(DeltaTimingCallback())
    trainer.callbacks.append(get_overlap_callback(cfg))
    trainer.callbacks.append(
        GarbageCollectionCallback(
            gc_interval_train=cfg.model.gc_interval_train,
            gc_interval_val=cfg.model.gc_interval_valid,
        )
    )
    if cfg.misc.print_config:
        trainer.callbacks.append(PrintArtifacts(cfg.exp_manager.explicit_log_dir))

    if cfg.model.nsys_profile.enabled:
        trainer.callbacks.append(
            NsysCallback(
                start_step=cfg.model.nsys_profile.start_step,
                end_step=cfg.model.nsys_profile.end_step,
                ranks=cfg.model.nsys_profile.ranks,
                gen_shape=cfg.model.nsys_profile.gen_shape,
                nvtx_ranges=cfg.model.nsys_profile.nvtx_ranges
            )
        )

    metrics_logger = MetricsLogger(
        model,
        cfg,
        optimizer,
    )

    custom_callback = metrics_logger.callback
    metrics_logger.set_trainer(trainer)

    trainer.loggers.append(metrics_logger)
    trainer.callbacks.append(custom_callback)

    memory_profiler = None
    if cfg.misc.memory_profiler.enable:
        memory_profiler = MemoryProfileCallback(
            file_prefix=cfg.misc.memory_profiler.file_prefix,
            max_entries=cfg.misc.memory_profiler.max_entries,
            rank_0_only=cfg.misc.memory_profiler.rank_0_only,
            start_location=cfg.misc.memory_profiler.start_location,
            end_location=cfg.misc.memory_profiler.end_location,
            force_oom_before_stop=cfg.misc.memory_profiler.force_oom_before_stop,
        )
        trainer.callbacks.append(memory_profiler)

    if fname := os.environ.get('STAT_CALLBACK_FNAME'):
        from mlperf_common.callbacks import StatsLogCallback
        trainer.callbacks.append(StatsLogCallback(save_path=fname))

    setup_auxiliary_loggers()
    log_hyperparams(cfg)

    # Setup warmups
    trainer.mock_dataset = warmup_data

    logging.info(f"======== Benchmarked fit ========")

    if cfg.misc.memory_profiler.enable and cfg.misc.memory_profiler.possible_oom:
        try:
            train(
                model=model,
                trainer=trainer,
                optim=optimizer,
                data=data,
                tokenizer="data",
                log=logger,
                resume=resume,
            )
        except:
            memory_profiler.cleanup_after_oom()
    else:
        train(
            model=model,
            trainer=trainer,
            optim=optimizer,
            data=data,
            tokenizer="data",
            log=logger,
            resume=resume,
        )


if __name__ == "__main__":
    if get_rank() == 0:
        mllogger.start(key=mllogger.constants.INIT_START)
    main()
