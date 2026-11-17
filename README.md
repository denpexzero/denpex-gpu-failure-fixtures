# Denpex Public GPU Failure Fixtures (25 Reproducible Cases)

A lightweight, public corpus subset of 25 real GPU training, distributed collective, and inference failure signatures extracted from open upstream issue trackers (PyTorch, DeepSpeed, vLLM, NCCL, and Hugging Face ecosystems).

The full scored benchmark suite and evaluation report live at [denpex.com/benchmark](https://denpex.com/benchmark), and you can test raw stack traces offline in your browser at [denpex.com/tools/gpu-log-analyzer](https://denpex.com/tools/gpu-log-analyzer).

## What Is Inside

- `fixtures/corpus-sample-25.json`: 25 anonymized public incident records with upstream `source_url`, `framework`, `failure_type` label, and exact raw `error_message` excerpt.
- `verify_fixtures.py`: Standalone Python 3 verification script that checks JSON schema integrity, duplicate IDs, non-empty error strings, and HTTPS source links.

## Quick Verification

```bash
python3 verify_fixtures.py
```

## Fixture Index

| Fixture ID | Failure Label | Framework | Summary | Upstream Source |
|---|---|---|---|---|
| `lf_github_issue_2873` | `STATE_DICT_SHAPE_MISMATCH` | FSDP | Error running distributed Qwen3-32B lora | [Source](https://github.com/meta-pytorch/torchtune/issues/2873) |
| `lf_github_issue_2830` | `OOM_FRAGMENTATION` | Axolotl | OOM handling and recovery | [Source](https://github.com/meta-pytorch/torchtune/issues/2830) |
| `lf_github_issue_45663` | `FSDP_ERROR` | DeepSpeed | Gemma-4 training with FSDP2 raises KeyError in Gemma4TextAttention.forward because shared_kv_states is rebuilt per-layer | [Source](https://github.com/huggingface/transformers/issues/45663) |
| `lf_github_issue_44928` | `GRADIENT_EXPLOSION` | DeepSpeed | Bug: Catastrophic gradient explosion (NaN) in RLHF with Qwen3.5 due to 3D position_ids forcing SDPA Math fallback and BF16 collapse | [Source](https://github.com/huggingface/transformers/issues/44928) |
| `lf_github_issue_43541` | `TORCH_COMPILE_DYNAMO` | DeepSpeed | RuntimeError: MixtralForCausalLM float32 model errors at grouped_mm op during torch dynamo tracing | [Source](https://github.com/huggingface/transformers/issues/43541) |
| `lf_github_issue_3169` | `TENSOR_DEVICE_MISMATCH` | Unknown | LoRA + BnB INT8 + CPU offload: output tensor on wrong device in tuners/lora/bnb.py | [Source](https://github.com/huggingface/peft/issues/3169) |
| `lf_github_issue_7844` | `SILENT_HANG` | DeepSpeed | Bug: DeepSpeed ZeRO-3 deadlock in engine.step() at step 0 under 2-GPU execution (RTX 3090, torch 2.2.1, DS 0.14.2) | [Source](https://github.com/deepspeedai/DeepSpeed/issues/7844) |
| `lf_github_issue_7811` | `DEEPSPEED_ERROR` | DeepSpeed | Bug: ZeRO-3 zero.GatheredParameters([multiple params], modifier_rank=None) + in-place slice touch triggers assert in free_param() | [Source](https://github.com/deepspeedai/DeepSpeed/issues/7811) |
| `lf_github_issue_7746` | `VERSION_MISMATCH` | DeepSpeed | Bug: Dtype mismatch (bf16 vs fp32) when resuming Muon optimizer from checkpoint | [Source](https://github.com/deepspeedai/DeepSpeed/issues/7746) |
| `lf_github_issue_7584` | `CHECKPOINT_CORRUPTION` | DeepSpeed | Bug: universal checkpoint for stage3 fails when there are multiple subgroups | [Source](https://github.com/deepspeedai/DeepSpeed/issues/7584) |
| `lf_github_issue_7549` | `DEEPSPEED_NVME_CHECKPOINT_RACE` | DeepSpeed | Bug: save_checkpoint race when consolidating NVMe offloaded tensors causing FileExistsError | [Source](https://github.com/deepspeedai/DeepSpeed/issues/7549) |
| `lf_github_issue_186357` | `TORCH_COMPILE_AOTI` | PyTorch | AOTInductor on Windows: models with >2GB constants fail to load with 32-bit lseek overflow | [Source](https://github.com/pytorch/pytorch/issues/186357) |
| `lf_github_issue_185535` | `TORCH_COMPILE_INDUCTOR` | PyTorch | Inductor cat lowering AssertionError with 1-D empty tensor and negative dim | [Source](https://github.com/pytorch/pytorch/issues/185535) |
| `lf_github_search_139884` | `PINNED_HOST_MEMORY_ALLOCATION_LIMIT` | PyTorch | Persistent memory leak from failed pinned memory allocation | [Source](https://github.com/pytorch/pytorch/issues/139884) |
| `lf_github_search_14766` | `NAN_LOSS` | PyTorch | NaN when training LayoutLM_V2 Model | [Source](https://github.com/huggingface/transformers/issues/14766) |
| `gc6_gc6-pytorch-pytorch-143580` | `DDP_ERROR` | pytorch | RuntimeError: expect_autograd_hooks_ INTERNAL ASSERT FAILED at c10d/reducer.cpp | [Source](https://github.com/pytorch/pytorch/issues/143580) |
| `gc6_gc6-deepspeedai-deepspeed-3574` | `DEEPSPEED_ZERO3_PARTITIONED_CHECKPOINT_SHAPE_MISMATCH` | deepspeed | Fail to load checkpoints after ZeRO-3 initialize | [Source](https://github.com/deepspeedai/DeepSpeed/issues/3574) |
| `depth_pytorch_190964` | `DEVICE_ASSERT` | pytorch | Illegal memory access for torch.prod with unusual sizes | [Source](https://github.com/pytorch/pytorch/issues/190964) |
| `lf_github_issue_43856` | `OOM_FRAGMENTATION` | DeepSpeed | Inefficient memory usage during Qwen3 MoE training | [Source](https://github.com/huggingface/transformers/issues/43856) |
| `lf_github_issue_3265` | `OOM_FRAGMENTATION` | FSDP | prepare_model_for_kbit_training adds ~1 GB CUDA reserved memory in 500 ms breaking 8 GB devices | [Source](https://github.com/huggingface/peft/issues/3265) |
| `lf_github_issue_3073` | `GRADIENT_EXPLOSION` | Unknown | LoRA gradients not normalized by input norm causing training instability (NaN) | [Source](https://github.com/huggingface/peft/issues/3073) |
| `lf_github_issue_7741` | `SILENT_HANG` | DeepSpeed | Bug: DecoupledCheckpointEngine hangs due to missing timeouts and process health checks | [Source](https://github.com/deepspeedai/DeepSpeed/issues/7741) |
| `lf_github_issue_7635` | `VERSION_MISMATCH` | DeepSpeed | Bug: TypeError DeepSpeedEngine.load_checkpoint() got an unexpected keyword argument weights_only | [Source](https://github.com/deepspeedai/DeepSpeed/issues/7635) |
| `lf_github_issue_7607` | `VERSION_MISMATCH` | DeepSpeed | Bug: ZeRO-2 ipg_buckets KeyError | [Source](https://github.com/deepspeedai/DeepSpeed/issues/7607) |
| `lf_github_issue_7581` | `DEEPSPEED_ERROR` | DeepSpeed | Bug: Unit Test TestFp8ComposabilityAcrossZero.test[fp16] failure | [Source](https://github.com/deepspeedai/DeepSpeed/issues/7581) |

## Scope and Attribution

- Every record in `fixtures/corpus-sample-25.json` cites its public upstream issue URL.
- This repository provides benchmark input fixtures only. Proprietary triage engines and classifier internals remain separate on [denpex.com](https://denpex.com).
