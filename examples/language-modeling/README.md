### llama-2 7b HuggingFace

#### Source
```
git clone https://github.com/huggingface/transformers.git
```

#### Steps

The cloned version of `run_clm.py` needs to have a few modifications
before it can run to remove references to packages not installed in
the container, etc. Other changes are also done to enable using
FlashAttention2 following instructions from:

https://huggingface.co/docs/transformers/perf_infer_gpu_one

The following diff command shows the changes made
to the original version to create the modified version used in this
example.

```
diff --git a/examples/pytorch/language-modeling/run_clm.py b/examples/pytorch/language-modeling/run_clm.py
index 794bb5f1c..82118f64b 100755
--- a/examples/pytorch/language-modeling/run_clm.py
+++ b/examples/pytorch/language-modeling/run_clm.py
@@ -45,17 +45,18 @@ from transformers import (
     Trainer,
     TrainingArguments,
     default_data_collator,
-    is_torch_xla_available,
+    is_torch_tpu_available,
+#    is_torch_xla_available,
     set_seed,
 )
-from transformers.testing_utils import CaptureLogger
+#from transformers.testing_utils import CaptureLogger
 from transformers.trainer_utils import get_last_checkpoint
 from transformers.utils import check_min_version, send_example_telemetry
 from transformers.utils.versions import require_version
 
 
 # Will error if the minimal version of Transformers is not installed. Remove at your own risks.
-check_min_version("4.45.0.dev0")
+#check_min_version("4.45.0.dev0")
 
 require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
 
@@ -149,6 +150,12 @@ class ModelArguments:
             )
         },
     )
+    use_flash_attention_2: bool = field(
+        default=True
+    )
+    device_map: Optional[str] = field(
+        default='auto'
+    )
 
     def __post_init__(self):
         if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
@@ -312,7 +319,7 @@ def main():
             cache_dir=model_args.cache_dir,
             token=model_args.token,
             streaming=data_args.streaming,
-            trust_remote_code=model_args.trust_remote_code,
+#            trust_remote_code=model_args.trust_remote_code,
         )
         if "validation" not in raw_datasets.keys():
             raw_datasets["validation"] = load_dataset(
@@ -434,6 +441,7 @@ def main():
             trust_remote_code=model_args.trust_remote_code,
             torch_dtype=torch_dtype,
             low_cpu_mem_usage=model_args.low_cpu_mem_usage,
+            use_flash_attention_2=True,
         )
     else:
         model = AutoModelForCausalLM.from_config(config, trust_remote_code=model_args.trust_remote_code)
@@ -458,14 +466,15 @@ def main():
     tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")
 
     def tokenize_function(examples):
-        with CaptureLogger(tok_logger) as cl:
-            output = tokenizer(examples[text_column_name])
-        # clm input could be much much longer than block_size
-        if "Token indices sequence length is longer than the" in cl.out:
-            tok_logger.warning(
-                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
-                " before being passed to the model."
-            )
+        # with CaptureLogger(tok_logger) as cl:
+        #     output = tokenizer(examples[text_column_name])
+        # # clm input could be much much longer than block_size
+        # if "Token indices sequence length is longer than the" in cl.out:
+        #     tok_logger.warning(
+        #         "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
+        #         " before being passed to the model."
+        #     )
+        output = tokenizer(examples[text_column_name])
         return output
 
     with training_args.main_process_first(desc="dataset map tokenization"):
@@ -589,10 +598,14 @@ def main():
         tokenizer=tokenizer,
         # Data collator will default to DataCollatorWithPadding, so we change it.
         data_collator=default_data_collator,
-        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_xla_available() else None,
+        compute_metrics=compute_metrics if training_args.do_eval and not is_torch_tpu_available() else None,
         preprocess_logits_for_metrics=preprocess_logits_for_metrics
-        if training_args.do_eval and not is_torch_xla_available()
+        if training_args.do_eval and not is_torch_tpu_available()
         else None,
+        # compute_metrics=compute_metrics if training_args.do_eval and not is_torch_xla_available() else None,
+        # preprocess_logits_for_metrics=preprocess_logits_for_metrics
+        # if training_args.do_eval and not is_torch_xla_available()
+        # else None,
     )
 
     # Training
@@ -648,9 +661,9 @@ def main():
         trainer.create_model_card(**kwargs)
 
 
-def _mp_fn(index):
-    # For xla_spawn (TPUs)
-    main()
+# def _mp_fn(index):
+#     # For xla_spawn (TPUs)
+#     main()
 
 
 if __name__ == "__main__":
```

A command similar to the following can be used to run the modified
`run_clm.py`. Note that you must provide your own HuggingFace token to
enable pulling the model, etc, in the command below in place of
`--token=<YOUR_HF_TOKEN_HERE>`. Further, the `ds-wrapper.sh` mentioned below
is a simple wrapper that defines a few relevant `HF` environment variables
and then executes the program passed to it. Some of the variables the
example `ds-wrapper.sh` set are:

```
export HF_DATASETS_CACHE=/lustre/scratch/username/hfcache
export TRANSFORMERS_CACHE=/lustre/scratch/username/hfcache
export HF_MODULES_CACHE=/lustre/scratch/username/hfcache
export HF_DATASETS_IN_MEMORY_MAX_SIZE=200000000
```

The example below runs on two nodes and uses four GPUs per node.

##### SS Containerized Libraries

```
$> env MASTER_ADDR=`scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1` srun --mpi=pmix_v4 -n 8 --ntasks-per-node=4 --distribution=*:block --cpu-bind=socket --ntasks-per-socket=1 -c 72 singularity run --nv --bind $TMPDIR --bind $HOME --bind `pwd` /path/to/sif/pytorch-ngc-hpc-dev.sif /projects/benchmarking/public/examples/llama-2/ds-wrapper.sh python /projects/benchmarking/public/examples/llama-2/transformers/examples/pytorch/language-modeling/run_clm.py --model_name_or_path meta-llama/Llama-2-7b-hf --dataset_name wikitext --dataset_config_name wikitext-2-v1 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --do_train --do_eval --output_dir `pwd`/output --overwrite_output_dir --token=<YOUR_HF_TOKEN_HERE> --block_size 4096 --torch_dtype=bfloat16 --bf16=True --deepspeed=/projects/benchmarking/public/examples/llama-2/ds_config.json --gradient_checkpointing=True >& out-n8-ppn4.txt
```

##### Bind-mount SS Libraries

```
$> env MASTER_ADDR=`scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1` srun --mpi=pmi2 -n 8 --ntasks-per-node=4 --distribution=*:block --cpu-bind=socket --ntasks-per-socket=1 -c 72 singularity run --nv --bind /projects --bind $TMPDIR --bind $HOME --bind `pwd` --bind /opt/cray/libfabric/1.15.2.0/lib64:/host/lib,/usr:/host/usr /projects/benchmarking/public/sif/pytorch-ngc-hpc-dev.sif /projects/benchmarking/public/examples/llama-2/ds-wrapper.sh python /projects/benchmarking/public/examples/llama-2/transformers/examples/pytorch/language-modeling/run_clm.py --model_name_or_path meta-llama/Llama-2-7b-hf --dataset_name wikitext --dataset_config_name wikitext-2-v1 --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --do_train --do_eval --output_dir `pwd`/output --overwrite_output_dir --token=<YOUR_HF_TOKEN_HERE> --block_size 4096 --torch_dtype=bfloat16 --bf16=True --deepspeed=/projects/benchmarking/public/examples/llama-2/ds_config.json --gradient_checkpointing=True >& out-n8-ppn4.txt
```

#### Example Output

Using the command above should give output similar to:

```
***** train metrics *****
  epoch                    =        3.0
  train_loss               =     1.6149
  train_runtime            = 0:02:57.89
  train_samples            =        679
  train_samples_per_second =      11.45
  train_steps_per_second   =      0.185
```

An estimated TFLOPs can be calculated using the following formula:

```
(train_samples_per_second * 6 * model_size * sequence_length)
```

In the case of llama-2-7b used above the model size is 7b, the sequence_length from above was 4096, and the 6
represents number of operations per weight, which gives:

```
11.45 * 6 * 7b * 4096 / 1t ==> (11.45 * 6 * 7 * 4096) / 1000 = 1969.7664 TFLOPs
```

For the TFLOPs per GPU simply divide the total by the number of GPUs used. For this run there were 8 total GPUs on
the 2 nodes:

```
((11.45 * 6 * 7 * 4096) / 1000) / 8) = 246.2 TFOPs per GPU
```

Note that increasing the `--per_device_train_batch_size` to 16 got closer to 262 TFLOPs per GPU and scaled well up to
8 nodes. The following are sample TFLOPs seen running with a batch size of 16 on 1 -> 8 nodes with 4 GPUs per node:

```
out-n4-ppn4.txt:  train_samples_per_second =      5.864
out-n8-ppn4.txt:  train_samples_per_second =     12.229
out-n16-ppn4.txt:  train_samples_per_second =     24.423
out-n32-ppn4.txt:  train_samples_per_second =     48.651
```

which equates to:
```
N1xppn4: 252.2 TFLOP/s per GPU
N2xppn4: 263.0 TFLOP/s per GPU
N4xppn4: 262.6 TFLOP/s per GPU
N8xppn4: 261.5 TFLOP/s per GPU
```

The runs on a single node had memory pressure causing more torch
memory flushes which impacted performance.

