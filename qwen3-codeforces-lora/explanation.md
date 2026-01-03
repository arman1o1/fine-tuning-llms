# Fine-tuning Qwen3-0.6B: Educational Guide

A step-by-step explanation of the `train_qwen3_sft.py` script for learning purposes.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Concepts](#key-concepts)
3. [Code Walkthrough](#code-walkthrough)
4. [Training Process](#training-process)
5. [Common Issues & Solutions](#common-issues--solutions)

---

## Overview

This script fine-tunes **Qwen3-0.6B** (a 600M parameter language model) on competitive programming problems using:

- **SFT (Supervised Fine-Tuning)** - Learning from input-output examples
- **LoRA (Low-Rank Adaptation)** - Efficient training by only updating small adapter layers

### Why These Choices?

| Choice | Reason |
|--------|--------|
| **Qwen3-0.6B** | Small enough to train quickly, large enough to be useful |
| **LoRA** | Only 1.67% of parameters trained (~10M vs 606M) |
| **bf16** | Faster training on modern GPUs |
| **SFT** | Dataset already has high-quality solutions |

---

## Key Concepts

### 1. LoRA (Low-Rank Adaptation)

Instead of updating all 606M parameters, LoRA adds small trainable matrices to specific layers:

```
Original: W (frozen, 606M params)
LoRA:     W + A×B (only A,B trained, ~10M params)
```

**Target modules in our script:**
- `q_proj, k_proj, v_proj, o_proj` → Attention layers
- `gate_proj, up_proj, down_proj` → MLP layers

**Key parameters:**
- `r=16` (rank) - Size of the low-rank matrices
- `lora_alpha=32` - Scaling factor (alpha/r = 2)
- `lora_dropout=0.05` - Regularization

### 2. Chat Template

The model expects conversations in a specific format. We use `apply_chat_template()` to convert:

```python
messages = [
    {"role": "system", "content": "You are an expert..."},
    {"role": "user", "content": "Solve this problem..."},
    {"role": "assistant", "content": "<think>...</think>code..."}
]
```

### 3. Gradient Checkpointing

Saves memory by recomputing activations during backward pass instead of storing them:

```python
gradient_checkpointing=True  # Trades compute for memory
```

---

## Code Walkthrough

### Section 1: Configuration (Lines 35-78)

```python
@dataclass
class Config:
    model_name: str = "Qwen/Qwen3-0.6B"
    lora_rank: int = 16
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4  # Effective batch = 16
```

**Why these values?**
- `learning_rate=2e-4` → Standard for LoRA fine-tuning
- `batch_size=4, accumulation=4` → Effective batch of 16 (memory efficient)

### Section 2: Dataset Preparation (Lines 96-181)

The dataset has two formats:
1. **Pre-formatted messages** - Use directly if available
2. **Raw fields** - Build from title, description, examples

```python
if "messages" in example:
    messages = example["messages"]
else:
    # Build from raw fields
    problem_text = title + description + input/output format
    messages = [system, user, assistant]
```

### Section 3: Model Loading (Lines 188-246)

```python
# 1. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.model_name)

# 2. Load model in bf16
model = AutoModelForCausalLM.from_pretrained(
    config.model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 3. Apply LoRA adapters
model = get_peft_model(model, lora_config)
```

### Section 4: Training (Lines 253-313)

```python
# SFTTrainer handles the training loop
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    processing_class=tokenizer,
)
trainer.train()
```

**Key SFTConfig settings:**
- `bf16=True` - Use bfloat16 precision
- `packing=False` - Don't pack multiple examples (simpler)
- `dataset_text_field="text"` - Column name with formatted text

### Section 5: Inference (Lines 395-467)

```python
# Load base model + LoRA adapter
base_model = AutoModelForCausalLM.from_pretrained(...)
model = PeftModel.from_pretrained(base_model, adapter_path)

# Generate with sampling
outputs = model.generate(
    max_new_tokens=4096,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)
```

---

## Training Process

### What Happens During Training

1. **Forward Pass**: Input → Model → Predicted tokens
2. **Loss Calculation**: Cross-entropy between predicted and actual tokens
3. **Backward Pass**: Compute gradients
4. **Update**: Only LoRA weights updated (1.67% of model)

### Key Metrics to Watch

| Metric | Good Sign | Bad Sign |
|--------|-----------|----------|
| `loss` | Decreasing | Flat or increasing |
| `grad_norm` | 0.2-2.0 | 0.0 (frozen) or >10 (exploding) |
| `token_accuracy` | Increasing | Flat |

### Expected Results

```
Start:  loss=2.0, accuracy=56%
End:    loss=0.75, accuracy=79%
Time:   ~26 minutes on MI300X
```

---

## Common Issues & Solutions

### 1. grad_norm = 0.0

**Problem:** Model not learning (gradients frozen)

**Solutions:**
- Disable 4-bit quantization (`load_in_4bit=False`)
- Check LoRA is applied correctly
- Verify `requires_grad=True` on LoRA params

### 2. Out of Memory

**Solutions:**
- Reduce `per_device_train_batch_size`
- Enable `gradient_checkpointing=True`
- Use 4-bit quantization (if gradients work)

### 3. Loss Not Decreasing

**Solutions:**
- Increase learning rate (try 5e-4)
- Check data formatting is correct
- Verify tokenizer has correct pad token

### 4. UnicodeEncodeError (Windows)

**Problem:** Emoji characters in print statements

**Solution:** Replace emojis with ASCII text (already done in this script)

---

## File Structure

```
project/
├── train_qwen3_sft.py      # Main training script
├── requirements_qwen3.txt   # Dependencies
├── README.md               # Project overview
├── explanation.md          # This file
├── .gitignore              # Git ignore rules
├── LICENSE                 # MIT License
└── outputs_qwen3_sft/      # Training outputs (git-ignored)
    ├── checkpoint-100/
    ├── checkpoint-200/
    └── final/              # Final trained model
```

---

## Quick Reference

### Train
```bash
python train_qwen3_sft.py
```

### Inference
```bash
python train_qwen3_sft.py --inference_only
```

### Push to Hub
```bash
python train_qwen3_sft.py --push_to_hub --hub_model_id USER/model-lora
```

---

## Further Learning

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [Qwen3 Model Card](https://huggingface.co/Qwen/Qwen3-0.6B)
