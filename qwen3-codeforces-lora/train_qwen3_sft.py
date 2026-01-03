#!/usr/bin/env python3
"""
Fine-tune Qwen3-0.6B on CodeForces-CoTS Dataset
Supervised Fine-Tuning with LoRA for Instruction Following

Usage:
    python train_qwen3_sft.py                          # Full training
    python train_qwen3_sft.py --max_steps 10           # Quick test
    python train_qwen3_sft.py --dry_run                # Validation run (10 steps, 50 samples)
    python train_qwen3_sft.py --inference_only         # Test inference
    
    # Push to Hugging Face Hub:
    python train_qwen3_sft.py --push_to_hub --hub_model_id USER/model-lora      # LoRA adapter only (~40MB)
    python train_qwen3_sft.py --merge_and_push --hub_model_id USER/model        # Merged full model (~1.2GB)

Requirements:
    pip install -r requirements_qwen3.txt
"""

import os
import argparse
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class Config:
    """Training configuration."""
    # Model
    model_name: str = "Qwen/Qwen3-0.6B"
    max_seq_length: int = 2048
    
    # LoRA
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    
    # Quantization - DISABLED for proper gradient flow
    # (4-bit was causing grad_norm=0.0)
    load_in_4bit: bool = False  # Disabled - you have 205GB VRAM
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    
    # Training
    learning_rate: float = 2e-4
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4  # Effective batch size = 4 x 4 = 16
    warmup_ratio: float = 0.1
    max_steps: int = -1  # -1 means use epochs
    
    # Dataset
    dataset_name: str = "open-r1/codeforces-cots"
    dataset_subset: str = "solutions_py"
    max_samples: Optional[int] = None  # None = use all
    
    # Output
    output_dir: str = "outputs_qwen3_sft"
    save_steps: int = 100
    logging_steps: int = 10
    
    # Misc
    seed: int = 42
    hf_token: Optional[str] = field(default_factory=lambda: os.environ.get("HF_TOKEN"))


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """You are an expert competitive programmer. When given a problem, provide:
1. Your reasoning and approach
2. A complete Python solution

Always think step-by-step before writing code."""


# ============================================================
# DATASET PREPARATION
# ============================================================

def format_example(example: dict, tokenizer) -> dict:
    """Format a dataset example as a chat conversation."""
    # The dataset has a 'messages' field that may already be formatted
    # If not, we use 'description' for problem and 'generation' for solution
    
    if "messages" in example and example["messages"]:
        # Use pre-formatted messages if available
        messages = example["messages"]
        if isinstance(messages, list) and len(messages) > 0:
            # Add system prompt if not present
            if messages[0].get("role") != "system":
                messages = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
    else:
        # Build from description and generation
        # Combine title, description, input/output format
        problem_parts = []
        if example.get("title"):
            problem_parts.append(f"# {example['title']}")
        if example.get("description"):
            problem_parts.append(example["description"])
        if example.get("input_format"):
            problem_parts.append(f"\n**Input:**\n{example['input_format']}")
        if example.get("output_format"):
            problem_parts.append(f"\n**Output:**\n{example['output_format']}")
        if example.get("examples"):
            problem_parts.append(f"\n**Examples:**\n{example['examples']}")
        
        problem_text = "\n\n".join(problem_parts)
        
        # Get the solution (contains <think>...</think> reasoning + code)
        solution = example.get("generation", "")
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Solve this competitive programming problem:\n\n{problem_text}"},
            {"role": "assistant", "content": solution}
        ]
    
    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    
    return {"text": text}


def load_and_prepare_dataset(config: Config, tokenizer):
    """Load and prepare the CodeForces-CoTS dataset."""
    print(f"\n[*] Loading dataset: {config.dataset_name}/{config.dataset_subset}")
    
    try:
        # Try loading the specific subset
        dataset = load_dataset(
            config.dataset_name,
            config.dataset_subset,
            split="train",
            token=config.hf_token,
        )
    except Exception as e:
        print(f"[!] Could not load subset '{config.dataset_subset}': {e}")
        print("Trying default configuration...")
        dataset = load_dataset(
            config.dataset_name,
            split="train",
            token=config.hf_token,
        )
    
    print(f"[OK] Loaded {len(dataset)} examples")
    print(f"   Columns: {dataset.column_names}")
    
    # Limit samples if specified
    if config.max_samples:
        dataset = dataset.select(range(min(config.max_samples, len(dataset))))
        print(f"   Using {len(dataset)} samples")
    
    # Format dataset
    print("[*] Formatting dataset...")
    dataset = dataset.map(
        lambda x: format_example(x, tokenizer),
        remove_columns=dataset.column_names,
        desc="Formatting"
    )
    
    return dataset


# ============================================================
# MODEL LOADING
# ============================================================

def load_model_and_tokenizer(config: Config):
    """Load model with bf16 precision and apply LoRA adapters."""
    print(f"\n[*] Loading model: {config.model_name}")
    
    # Quantization config
    bnb_config = None
    if config.load_in_4bit:
        compute_dtype = getattr(torch, config.bnb_4bit_compute_dtype)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=True,
        )
        print("   Using 4-bit quantization")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        token=config.hf_token,
    )
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        token=config.hf_token,
        torch_dtype=torch.bfloat16 if not config.load_in_4bit else None,
    )
    
    print(f"[OK] Model loaded on {model.device}")
    
    # Prepare for k-bit training
    if config.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # LoRA config - target all attention and MLP projection layers
    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",  # Attention
                        "gate_proj", "up_proj", "down_proj"],    # MLP
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model, tokenizer


# ============================================================
# TRAINING
# ============================================================

def train(config: Config):
    """Main training function."""
    print("=" * 60)
    print(">>> Qwen3-0.6B Fine-tuning on CodeForces-CoTS")
    print("=" * 60)
    
    # GPU info
    if torch.cuda.is_available():
        print(f"\n[GPU] {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\n[!] No GPU detected, training will be slow!")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config)
    
    # Load dataset
    dataset = load_and_prepare_dataset(config, tokenizer)
    
    # Training arguments - compatible with trl 0.26.2
    training_args = SFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        max_steps=config.max_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=2,
        fp16=False,
        bf16=True,  # Enable bf16 for MI300X
        packing=False,
        dataset_text_field="text",
        seed=config.seed,
        report_to="none",
        optim="adamw_torch",
        remove_unused_columns=False,
        gradient_checkpointing=True,  # Save memory, ensure gradient flow
    )
    
    # Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    
    # Train!
    print("\n[*] Starting training...")
    trainer.train()
    
    # Save locally
    print(f"\n[*] Saving model to {config.output_dir}/final")
    trainer.save_model(f"{config.output_dir}/final")
    tokenizer.save_pretrained(f"{config.output_dir}/final")
    
    print("\n[OK] Training complete!")
    return model, tokenizer


def push_to_hub(config: Config, hub_model_id: str, merge: bool = False):
    """Push trained model to Hugging Face Hub.
    
    Args:
        config: Training configuration
        hub_model_id: Hugging Face repo ID (e.g., username/model-name)
        merge: If True, merge LoRA into base model before pushing
    """
    from huggingface_hub import HfApi, login, create_repo
    from peft import PeftModel
    
    adapter_path = f"{config.output_dir}/final"
    
    if not os.path.exists(adapter_path):
        print(f"[ERROR] No trained model found at {adapter_path}")
        return
    
    # Login if token is available
    if config.hf_token:
        login(token=config.hf_token)
    
    # Create repo if it doesn't exist
    print(f"\n[*] Creating/checking repository: {hub_model_id}")
    create_repo(hub_model_id, repo_type="model", exist_ok=True)
    
    if merge:
        # Merge LoRA into base model
        print(f"\n[*] Merging LoRA adapter with base model...")
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Load and merge LoRA
        model = PeftModel.from_pretrained(base_model, adapter_path)
        model = model.merge_and_unload()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
        
        # Save merged model locally
        merged_path = f"{config.output_dir}/merged"
        print(f"[*] Saving merged model to {merged_path}")
        model.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)
        
        # Push merged model
        print(f"\n[*] Pushing MERGED model to: {hub_model_id}")
        api = HfApi()
        api.upload_folder(
            folder_path=merged_path,
            repo_id=hub_model_id,
            repo_type="model",
            commit_message="Upload Qwen3-0.6B fine-tuned on CodeForces-CoTS (merged)",
        )
        print(f"[OK] Merged model pushed to: https://huggingface.co/{hub_model_id}")
        
    else:
        # Push LoRA adapter only
        print(f"\n[*] Pushing LoRA adapter to: {hub_model_id}")
        api = HfApi()
        api.upload_folder(
            folder_path=adapter_path,
            repo_id=hub_model_id,
            repo_type="model",
            commit_message="Upload Qwen3-0.6B LoRA adapter fine-tuned on CodeForces-CoTS",
        )
        print(f"[OK] LoRA adapter pushed to: https://huggingface.co/{hub_model_id}")
        print(f"    Base model required: {config.model_name}")


# ============================================================
# INFERENCE
# ============================================================

def run_inference(config: Config, prompt: Optional[str] = None):
    """Run inference with a trained model."""
    from peft import PeftModel
    
    model_path = f"{config.output_dir}/final"
    
    if not os.path.exists(model_path):
        print(f"[ERROR] No trained model found at {model_path}")
        print("   Please train first: python train_qwen3_sft.py")
        return
    
    print(f"\n[*] Loading trained model from {model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # Load base model (use bf16 to match training)
    base_model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    # Load LoRA weights
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    # Default test prompt
    if not prompt:
        prompt = """You are given an array of n integers. Find the maximum sum of any contiguous subarray.

Input: First line contains n. Second line contains n integers.
Output: Print the maximum subarray sum.

Example:
Input:
9
-2 1 -3 4 -1 2 1 -5 4
Output:
6

Constraints: 1 <= n <= 10^5, -10^4 <= a[i] <= 10^4"""
    
    print(f"\n[Problem]\n{prompt[:200]}...")
    
    # Format as chat
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Solve this competitive programming problem:\n\n{prompt}"}
    ]
    
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    print("\n[*] Generating solution...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096,  # Increased for long reasoning chains
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\n[Solution]\n{response}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-0.6B on CodeForces-CoTS")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max training steps (-1 for epochs)")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit dataset size")
    parser.add_argument("--inference_only", action="store_true", help="Run inference only")
    parser.add_argument("--prompt", type=str, default=None, help="Custom prompt for inference")
    parser.add_argument("--output_dir", type=str, default="outputs_qwen3_sft", help="Output directory")
    parser.add_argument("--dry_run", action="store_true", help="Quick validation run")
    parser.add_argument("--push_to_hub", action="store_true", help="Push LoRA adapter to Hugging Face Hub")
    parser.add_argument("--merge_and_push", action="store_true", help="Merge LoRA with base model and push full model")
    parser.add_argument("--hub_model_id", type=str, default=None, help="Hugging Face repo ID (e.g., username/model-name)")
    args = parser.parse_args()
    
    # Create config
    config = Config()
    config.output_dir = args.output_dir
    
    if args.max_steps > 0:
        config.max_steps = args.max_steps
    
    if args.max_samples:
        config.max_samples = args.max_samples
    
    if args.dry_run:
        config.max_steps = 10
        config.max_samples = 50
        config.logging_steps = 1
        print("[DRY RUN] Quick validation with 10 steps")
    
    if args.inference_only:
        run_inference(config, args.prompt)
    elif args.push_to_hub or args.merge_and_push:
        if not args.hub_model_id:
            print("[ERROR] --hub_model_id is required when pushing to Hub")
            print("   Example: --push_to_hub --hub_model_id username/qwen3-codeforces-lora")
            print("   Example: --merge_and_push --hub_model_id username/qwen3-codeforces")
            return
        push_to_hub(config, args.hub_model_id, merge=args.merge_and_push)
    else:
        train(config)


if __name__ == "__main__":
    main()
