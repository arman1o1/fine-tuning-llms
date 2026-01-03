# Qwen3-0.6B Fine-tuning on CodeForces-CoTS

Fine-tune [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) on the [CodeForces-CoTS](https://huggingface.co/datasets/open-r1/codeforces-cots) dataset for competitive programming.

## Features

- **SFT + LoRA** - Efficient fine-tuning with only 1.67% trainable parameters
- **bf16 Training** - Optimized for modern GPUs
- **Easy Push to Hub** - Upload to Hugging Face with one command
- **Inference Ready** - Test your model immediately after training

## Quick Start

### Installation

```bash
pip install -r requirements_qwen3.txt
```

### Training

```bash
# Full training (3 epochs on 9.5k examples)
python train_qwen3_sft.py

# Quick test (10 steps)
python train_qwen3_sft.py --dry_run

# Custom training
python train_qwen3_sft.py --max_samples 1000 --max_steps 500
```

### Inference

```bash
python train_qwen3_sft.py --inference_only
```

### Push to Hugging Face

```bash
# Login first
huggingface-cli login

# Push LoRA adapter (~40MB)
python train_qwen3_sft.py --push_to_hub --hub_model_id YOUR_USERNAME/qwen3-codeforces-lora

# Push merged model (~1.2GB)
python train_qwen3_sft.py --merge_and_push --hub_model_id YOUR_USERNAME/qwen3-codeforces
```

## Training Results

| Metric | Start | End | Change |
|--------|-------|-----|--------|
| Loss | 2.04 | 0.75 | -63% |
| Token Accuracy | 56% | 79% | +23% |

Training time: ~26 minutes on AMD MI300X (205GB VRAM)

## Model Architecture

- **Base Model**: Qwen/Qwen3-0.6B (606M parameters)
- **LoRA Rank**: 16
- **Trainable Parameters**: 10M (1.67%)
- **Target Modules**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

## Dataset

[open-r1/codeforces-cots](https://huggingface.co/datasets/open-r1/codeforces-cots) - 9,556 competitive programming problems with chain-of-thought solutions.

## License

MIT
