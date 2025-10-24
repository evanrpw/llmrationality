#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
import json
import numpy as np
import wandb
import argparse
import os
from pathlib import Path


class CoherenceDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512, split='train', val_split=0.2, random_seed=42):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        # Keep pairs together - each item contains both statement and negation
        all_pairs = []
        for item in data:
            all_pairs.append({
                'statement': item['proposition_correct_answer'],
                'negation': item['negation_correct_answer'],
                'item_id': item['id']
            })
        
        # Split data into train/validation
        import random
        random.seed(random_seed)
        random.shuffle(all_pairs)
        
        split_idx = int(len(all_pairs) * (1 - val_split))
        
        if split == 'train':
            self.pairs = all_pairs[:split_idx]
        elif split == 'val':
            self.pairs = all_pairs[split_idx:]
        else:
            raise ValueError(f"Split must be 'train' or 'val', got {split}")
    
    def __len__(self):
        return len(self.pairs)
    
    def _create_prompt(self, text):
        """Create prompt for a single text"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f'Statement: "{text}"\nIs the statement true or false? Please answer in a single word, either True or False, with no other output.'},
            # {"role": "assistant", "content": 'The statement is '}
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    
    def compute_original_confidences(self, model):
        """Compute original model confidence scores for statements and negations"""
        model.eval()
        device = next(model.parameters()).device
        
        for idx in range(len(self.pairs)):
            pair = self.pairs[idx]
            statement_prompt = self._create_prompt(pair['statement'])
            negation_prompt = self._create_prompt(pair['negation'])
            
            statement_encoding = create_tokenized_encoding(self.tokenizer, statement_prompt, self.max_length)
            negation_encoding = create_tokenized_encoding(self.tokenizer, negation_prompt, self.max_length)
            
            # Move to device
            statement_input_ids = statement_encoding['input_ids'].to(device)
            statement_attention_mask = statement_encoding['attention_mask'].to(device)
            negation_input_ids = negation_encoding['input_ids'].to(device)
            negation_attention_mask = negation_encoding['attention_mask'].to(device)
            
            with torch.no_grad():
                stmt_true_prob, _, _ = get_true_false_probabilities(
                    model, self.tokenizer, statement_input_ids, statement_attention_mask
                )
                neg_true_prob, _, _ = get_true_false_probabilities(
                    model, self.tokenizer, negation_input_ids, negation_attention_mask
                )
            
            pair['original_confidence'] = stmt_true_prob / (stmt_true_prob + neg_true_prob)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        # Create prompts for both statement and negation
        statement_prompt = self._create_prompt(pair['statement'])
        negation_prompt = self._create_prompt(pair['negation'])
        
        # Tokenize both using helper function
        statement_encoding = create_tokenized_encoding(self.tokenizer, statement_prompt, self.max_length)
        negation_encoding = create_tokenized_encoding(self.tokenizer, negation_prompt, self.max_length)
        
        return {
            'statement_input_ids': statement_encoding['input_ids'].squeeze(),
            'statement_attention_mask': statement_encoding['attention_mask'].squeeze(),
            'negation_input_ids': negation_encoding['input_ids'].squeeze(),
            'negation_attention_mask': negation_encoding['attention_mask'].squeeze(),
            'item_id': pair['item_id'],
            'statement_text': pair['statement'],
            'negation_text': pair['negation'],
            'original_confidence': pair['original_confidence']
        }


def collate_fn(batch):
    """Collate function for paired data"""
    return {
        'statement_input_ids': torch.stack([item['statement_input_ids'] for item in batch]),
        'statement_attention_mask': torch.stack([item['statement_attention_mask'] for item in batch]),
        'negation_input_ids': torch.stack([item['negation_input_ids'] for item in batch]),
        'negation_attention_mask': torch.stack([item['negation_attention_mask'] for item in batch]),
        'item_id': [item['item_id'] for item in batch],
        'statement_text': [item['statement_text'] for item in batch],
        'negation_text': [item['negation_text'] for item in batch],
        'original_confidence': torch.tensor([item['original_confidence'] for item in batch])
    }


def move_batch_to_device(batch, device):
    """Move tensor fields in batch to specified device"""
    batch['statement_input_ids'] = batch['statement_input_ids'].to(device)
    batch['statement_attention_mask'] = batch['statement_attention_mask'].to(device)
    batch['negation_input_ids'] = batch['negation_input_ids'].to(device)
    batch['negation_attention_mask'] = batch['negation_attention_mask'].to(device)
    return batch


def save_checkpoint(model, optimizer, epoch, global_step, config, metrics, checkpoint_dir):
    """Save model checkpoint with training state"""
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-epoch-{epoch + 1}")
    os.makedirs(checkpoint_path, exist_ok=True)
    
    # Save the PEFT model (LoRA adapters)
    model.save_pretrained(checkpoint_path)
    
    # Save optimizer state and training metadata
    checkpoint_data = {
        'epoch': epoch,
        'global_step': global_step,
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config,
        'metrics': metrics
    }
    
    torch.save(checkpoint_data, os.path.join(checkpoint_path, 'training_state.pt'))
    
    print(f"Checkpoint saved to {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, optimizer, tokenizer):
    """Load model checkpoint and restore training state"""
    print(f"Loading checkpoint from {checkpoint_path}")
    
    # Load the PEFT model
    model.load_adapter(checkpoint_path, adapter_name="default")
    
    # Load training state
    training_state_path = os.path.join(checkpoint_path, 'training_state.pt')
    if os.path.exists(training_state_path):
        checkpoint_data = torch.load(training_state_path, map_location='cpu')
        
        # Restore optimizer state
        optimizer.load_state_dict(checkpoint_data['optimizer_state_dict'])
        
        return {
            'epoch': checkpoint_data['epoch'],
            'global_step': checkpoint_data['global_step'],
            'config': checkpoint_data['config'],
            'metrics': checkpoint_data.get('metrics', {})
        }
    else:
        print("Warning: training_state.pt not found, starting from scratch")
        return {'epoch': -1, 'global_step': 0, 'config': {}, 'metrics': {}}


def find_latest_checkpoint(checkpoint_dir):
    """Find the most recent checkpoint in the directory"""
    if not os.path.exists(checkpoint_dir):
        return None
    
    checkpoints = []
    for item in os.listdir(checkpoint_dir):
        if item.startswith('checkpoint-epoch-'):
            try:
                epoch_num = int(item.split('-')[-1])
                checkpoints.append((epoch_num, os.path.join(checkpoint_dir, item)))
            except ValueError:
                continue
    
    if checkpoints:
        # Return path to most recent checkpoint
        return max(checkpoints, key=lambda x: x[0])[1]
    return None


def log_eval_metrics(metrics, epoch, improvement=None, wandb_enabled=False, prefix="eval"):
    """Log evaluation metrics to wandb with consistent naming"""
    if not wandb_enabled:
        return
    
    log_data = {
        f"{prefix}/mean_violation": metrics['mean_violation'],
        f"{prefix}/max_violation": metrics['max_violation'],
        f"{prefix}/median_violation": metrics['median_violation'],
        f"{prefix}/mean_sum": metrics['mean_sum'],
        "epoch": epoch
    }
    
    if improvement is not None:
        log_data[f"{prefix}/improvement"] = improvement
    
    # Create scatterplot if we have the data
    if (epoch % 4 == 0) and 'statement_prob_true' in metrics and 'negation_prob_true' in metrics:
        # Create scatterplot data
        scatter_data = []
        for p_stmt, p_neg in zip(metrics['statement_prob_true'], metrics['negation_prob_true']):
            scatter_data.append([p_stmt, p_neg])
        
        # Create wandb scatterplot
        scatter_table = wandb.Table(
            data=scatter_data,
            columns=["P(Statement=True)", "P(Negation=True)"]
        )
        
        dataset_type = "Validation" if prefix == "eval" else "Training"
        log_data[f"{prefix}/coherence_scatterplot_epoch_{epoch}"] = wandb.plot.scatter(
            scatter_table, 
            "P(Statement=True)", 
            "P(Negation=True)",
            title=f"Coherence: P(Statement=True) vs P(Negation=True) ({dataset_type} Set, Epoch {epoch})"
        )
        
    wandb.log(log_data)


def create_tokenized_encoding(tokenizer, prompt, max_length):
    """Create a tokenized encoding for a single prompt"""
    return tokenizer(
        prompt, truncation=True, padding='max_length', 
        max_length=max_length, return_tensors='pt'
    )


def get_true_false_probabilities(model, tokenizer, input_ids, attention_mask):
    """Get True/False probabilities from model output with optional debug info"""
    # Forward pass
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    
    # Get last token logits (padding is on left, so last token is at position -1)
    last_token_logits = logits[:, -1, :]  # Shape: [batch_size, vocab_size]
    
    # seq_lengths = attention_mask.sum(dim=1) - 1
    # batch_indices = torch.arange(logits.size(0), device=logits.device)
    # last_token_logits = logits[batch_indices, seq_lengths, :]
    

    # Get True/False token IDs
    true_token_id = tokenizer.encode('True', add_special_tokens=False)[0]
    false_token_id = tokenizer.encode('False', add_special_tokens=False)[0]
    
    # Extract True/False logits
    true_logits = last_token_logits[:, true_token_id]
    false_logits = last_token_logits[:, false_token_id]
    response_logits = torch.stack([true_logits, false_logits], dim=1)
    
    # Get probabilities, softmax over True/False to sum to 1
    probs = F.softmax(response_logits, dim=1)
    
    # return probs[:, 0], last_token_logits  # Return P(True) and logits for debug
    return probs[:, 0], probs[:, 1], last_token_logits # Return P(True), P(False), and logits for debug


def coherence_loss(model, tokenizer, batch):
    """Compute coherence loss for paired data using MSE on probability sum"""
    # Get probabilities for statements
    statement_true_probs, _, statement_logits = get_true_false_probabilities(
        model, tokenizer, 
        batch['statement_input_ids'], 
        batch['statement_attention_mask']
    )
    
    # Get probabilities for negations  
    negation_true_probs, _, negation_logits = get_true_false_probabilities(
        model, tokenizer,
        batch['negation_input_ids'],
        batch['negation_attention_mask']
    )
    
    # Sum of probabilities should be 1 (P(statement=True) + P(negation=True) = 1)
    prob_sums = statement_true_probs + negation_true_probs
    target = torch.ones_like(prob_sums)
    
    # MSE loss
    loss = F.mse_loss(prob_sums, target)

    return loss


def normalize_probabilities(statement_true_probs, negation_true_probs, use_softmax=True, temperature=2.0):
    """
    Normalize probabilities using either softmax or manual division normalization.
    
    Args:
        statement_true_probs: P(statement=True) probabilities
        negation_true_probs: P(negation=True) probabilities  
        use_softmax: If True, use softmax normalization; if False, use manual division
        temperature: Temperature parameter for softmax (higher = softer)
    
    Returns:
        Tuple of (normalized_statement_prob, normalized_negation_prob)
    """
    if use_softmax:
        # Softmax normalization with temperature
        # Higher temperature = softer targets?
        # Lower temperature = harder targets?
        coherence_probs = F.softmax(
            torch.log(torch.stack([statement_true_probs, negation_true_probs], dim=1)) / temperature, 
            dim=1
        )
        normalized_statement_prob = coherence_probs[:, 0]
        normalized_negation_prob = coherence_probs[:, 1]
    else:
        # Manual division normalization
        # This preserves the relative confidence ratios better than softmax
        prob_sums = statement_true_probs + negation_true_probs
        # Add small epsilon to avoid division by zero
        prob_sums = torch.clamp(prob_sums, min=1e-8)
        normalized_statement_prob = statement_true_probs / prob_sums
        normalized_negation_prob = negation_true_probs / prob_sums
    
    return normalized_statement_prob, normalized_negation_prob


def smooth_crossentropy_loss(model, tokenizer, batch, use_softmax=True, temperature=2.0):
    """
    Compute coherence loss using smooth cross-entropy approach.
    
    This approach:
    1. Gets P(statement=True) and P(negation=True) from the model
    2. Normalizes them so they sum to 1 (preserving relative confidence)
    3. Uses normalized probabilities as soft targets for cross-entropy loss
    4. Applies cross-entropy loss to both statement and negation predictions
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        batch: Batch of training data
        use_softmax: If True, use softmax normalization; if False, use manual division
        temperature: Temperature parameter for softmax normalization
    """
    # Get probabilities for statements
    statement_true_probs, _, statement_logits = get_true_false_probabilities(
        model, tokenizer, 
        batch['statement_input_ids'], 
        batch['statement_attention_mask']
    )
    
    # Get probabilities for negations  
    negation_true_probs, _, negation_logits = get_true_false_probabilities(
        model, tokenizer,
        batch['negation_input_ids'],
        batch['negation_attention_mask']
    )
    
    # # Step 1: Normalize probabilities using selected method
    # normalized_statement_prob, normalized_negation_prob = normalize_probabilities(
    #     statement_true_probs, negation_true_probs, use_softmax, temperature
    # )

    # Step 1: Take normalized original confidence as training target
    normalized_statement_prob = batch['original_confidence'].to(statement_logits.device)
    
    # Step 2: Create soft target distributions; ANCHOR ON TRUE
    # For statement: [normalized_statement_prob, 1 - normalized_statement_prob]
    # # For negation: [normalized_negation_prob, 1 - normalized_negation_prob]
    # For negation: [1 - normalized_statement_prob, normalized_statement_prob]
    statement_targets = torch.stack([
        normalized_statement_prob,  # P(True)
        1.0 - normalized_statement_prob  # P(False)
    ], dim=1)
    
    # negation_targets = torch.stack([
    #     normalized_negation_prob,   # P(True)  
    #     1.0 - normalized_negation_prob  # P(False)
    # ], dim=1)
    negation_targets = torch.stack([
        1.0 - normalized_statement_prob,
        normalized_statement_prob
    ], dim=1)
    
    # Step 3: Get model's current True/False distributions
    # We need to reconstruct the full True/False probability distributions
    true_token_id = tokenizer.encode('True', add_special_tokens=False)[0]
    false_token_id = tokenizer.encode('False', add_special_tokens=False)[0]
    
    # statement_logits and negation_logits are already the last token logits (2D: [batch_size, vocab_size])
    # Extract True/False logits directly
    statement_true_logits = statement_logits[:, true_token_id]
    statement_false_logits = statement_logits[:, false_token_id]
    statement_response_logits = torch.stack([statement_true_logits, statement_false_logits], dim=1)
    
    # For negations  
    negation_true_logits = negation_logits[:, true_token_id]
    negation_false_logits = negation_logits[:, false_token_id]
    negation_response_logits = torch.stack([negation_true_logits, negation_false_logits], dim=1)
    
    # Step 4: Apply cross-entropy loss with soft targets
    statement_loss = F.cross_entropy(
        statement_response_logits, 
        statement_targets,
        reduction='mean'
    )
    
    negation_loss = F.cross_entropy(
        negation_response_logits,
        negation_targets, 
        reduction='mean'
    )
    
    # # Step 4: Apply KL divergence loss with soft targets
    # statement_loss = F.kl_div(
    #     F.log_softmax(statement_response_logits, dim=1), 
    #     statement_targets,
    #     reduction='batchmean'
    # )

    # negation_loss = F.kl_div(
    #     F.log_softmax(negation_response_logits, dim=1),
    #     negation_targets, 
    #     reduction='batchmean'
    # )

    # Combine losses
    total_loss = (statement_loss + negation_loss) / 2
    
    return total_loss


def evaluate_coherence(model, tokenizer, dataset, batch_size=4):
    """Evaluate coherence violations on paired data"""
    model.eval()
    
    # Get device from model
    device = next(model.parameters()).device
    
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
    
    violations = []
    all_statement_probs = []
    all_negation_probs = []
    
    # For scatterplot: collect P(True) for statements and negations
    statement_prob_true = []
    negation_prob_true = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            # Move to device using helper function
            batch = move_batch_to_device(batch, device)
            
            # Get probabilities
            statement_true_probs, _, statement_logits = get_true_false_probabilities(
                model, tokenizer,
                batch['statement_input_ids'],
                batch['statement_attention_mask']
            )
            
            negation_true_probs, _, negation_logits = get_true_false_probabilities(
                model, tokenizer,
                batch['negation_input_ids'],
                batch['negation_attention_mask']
            )
            
            # Store probabilities for scatterplot
            for j in range(len(statement_true_probs)):
                statement_prob_true.append(statement_true_probs[j].item())
                negation_prob_true.append(negation_true_probs[j].item())
            
            # Debug output for first batch
            if i == 0:
                statement_topk_probs, statement_topk_indices = torch.topk(F.softmax(statement_logits, dim=1), 5, dim=1)
                negation_topk_probs, negation_topk_indices = torch.topk(F.softmax(negation_logits, dim=1), 5, dim=1)

                for j in range(min(3, len(statement_true_probs))):
                    stmt_prob = statement_true_probs[j].item()
                    neg_prob = negation_true_probs[j].item()
                    prob_sum = stmt_prob + neg_prob
                    violation = abs(prob_sum - 1.0)
                    
                    print(f"\nItem {batch['item_id'][j]}:")
                    print(f"  Statement: '{batch['statement_text'][j][:50]}...' -> P(True)={stmt_prob:.4f}")
                    
                    # Show top tokens for statement
                    stmt_topk_tokens = [tokenizer.decode([idx]) for idx in statement_topk_indices[j].tolist()]
                    print(f"    Top tokens: {[(tok.strip(), f'{prob:.3f}') for tok, prob in zip(stmt_topk_tokens, statement_topk_probs[j].tolist())]}")
                    
                    print(f"  Negation:  '{batch['negation_text'][j][:50]}...' -> P(True)={neg_prob:.4f}")
                    
                    # Show top tokens for negation
                    neg_topk_tokens = [tokenizer.decode([idx]) for idx in negation_topk_indices[j].tolist()]
                    print(f"    Top tokens: {[(tok.strip(), f'{prob:.3f}') for tok, prob in zip(neg_topk_tokens, negation_topk_probs[j].tolist())]}")
                    
                    print(f"  Sum: {prob_sum:.4f}, Violation: {violation:.4f}")
            
            prob_sums = statement_true_probs + negation_true_probs
            batch_violations = torch.abs(prob_sums - 1.0)

            for j in range(len(batch_violations)):
                violations.append(batch_violations[j].item())
                all_statement_probs.append(statement_true_probs[j].item())
                all_negation_probs.append(negation_true_probs[j].item())
    
    return {
        'mean_violation': np.mean(violations),
        'max_violation': np.max(violations),
        'median_violation': np.median(violations),
        'violations': violations,
        'mean_statement_prob': np.mean(all_statement_probs),
        'mean_negation_prob': np.mean(all_negation_probs),
        'mean_sum': np.mean(all_statement_probs) + np.mean(all_negation_probs),
        'statement_prob_true': statement_prob_true,
        'negation_prob_true': negation_prob_true
    }


def main():
    parser = argparse.ArgumentParser(description="Coherence Fine-tuning with paired data")
    parser.add_argument("--data_path", type=str, default="dataset.json", help="Path to training data")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="coherence-training", help="wandb project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="wandb run name")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model name")

    # Checkpoint arguments
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--auto_resume", action="store_true", help="Automatically resume from latest checkpoint")
    
    # Data split arguments
    parser.add_argument("--val_split", type=float, default=0.2, help="Fraction of data to use for validation")
    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for data splitting")
    parser.add_argument("--eval_train", action="store_true", help="Also evaluate on training set each epoch (for debugging overfitting)")
    
    # Loss function arguments
    parser.add_argument("--loss_type", type=str, default="mse", choices=["mse", "smooth_xent"], 
                       help="Loss function type: 'mse' for MSE on probability sums, 'smooth_xent' for smooth cross-entropy")
    parser.add_argument("--temperature", type=float, default=2.0, 
                       help="Temperature for softmax normalization (higher = softer targets, preserves confidence; only used with --use_softmax)")
    parser.add_argument("--use_softmax", action="store_true", 
                       help="Use softmax normalization instead of manual division. Softmax flattens differences but respects temperature, manual division preserves ratios exactly")
    args = parser.parse_args()
    
    # Validate arguments
    if not (0.0 < args.val_split < 1.0):
        raise ValueError(f"val_split must be between 0 and 1, got {args.val_split}")
    if args.temperature <= 0:
        raise ValueError(f"temperature must be positive, got {args.temperature}")
    
    # Configuration
    model_name = args.model_name

    config = {
        "model_name": model_name,
        "data_path": args.data_path,
        "num_epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "loss_type": args.loss_type,
        "temperature": args.temperature,
        "use_softmax": args.use_softmax,
        "val_split": args.val_split,
        "random_seed": args.random_seed,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]
    }
    
    # Setup checkpoint directory
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Determine resume checkpoint
    resume_checkpoint = None
    if args.resume_from:
        resume_checkpoint = args.resume_from
    elif args.auto_resume:
        resume_checkpoint = find_latest_checkpoint(checkpoint_dir)
        if resume_checkpoint:
            print(f"Auto-resuming from: {resume_checkpoint}")
    
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"], padding_side='left')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # Setup LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"]
    )
    model = get_peft_model(model, lora_config)
    
    # Create datasets and dataloaders
    train_dataset = CoherenceDataset(config["data_path"], tokenizer, split='train', 
                                   val_split=config["val_split"], random_seed=config["random_seed"])
    val_dataset = CoherenceDataset(config["data_path"], tokenizer, split='val', 
                                 val_split=config["val_split"], random_seed=config["random_seed"])
    
    # Compute original confidences for both datasets
    train_dataset.compute_original_confidences(model)
    val_dataset.compute_original_confidences(model)
    
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn)
    
    print(f"Training dataset size: {len(train_dataset)} pairs")
    print(f"Validation dataset size: {len(val_dataset)} pairs")
    print(f"Loss type: {config['loss_type']}")
    if config['loss_type'] == 'smooth_xent':
        norm_method = "softmax" if config['use_softmax'] else "manual division"
        print(f"Normalization method: {norm_method}")
        print(f"Temperature: {config['temperature']}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    
    # Initialize training state
    start_epoch = 0
    global_step = 0
    all_eval_metrics = []
    baseline_metrics = None
    
    # Load checkpoint if resuming
    if resume_checkpoint:
        checkpoint_data = load_checkpoint(resume_checkpoint, model, optimizer, tokenizer)
        start_epoch = checkpoint_data['epoch'] + 1
        global_step = checkpoint_data['global_step']
        print(f"Resuming from epoch {start_epoch}, global step {global_step}")
        
        # If we have saved metrics, use them
        if 'metrics' in checkpoint_data and 'baseline' in checkpoint_data['metrics']:
            baseline_metrics = checkpoint_data['metrics']['baseline']
            all_eval_metrics = checkpoint_data['metrics'].get('all_eval_metrics', [])
    
    # Initialize wandb
    if args.use_wandb:
        # Resume wandb run if we have a resume checkpoint
        wandb_id = None
        if resume_checkpoint and os.path.exists(os.path.join(resume_checkpoint, 'wandb_id.txt')):
            with open(os.path.join(resume_checkpoint, 'wandb_id.txt'), 'r') as f:
                wandb_id = f.read().strip()
        
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=config,
            resume="must" if wandb_id else None,
            id=wandb_id
        )
        
        # Save wandb ID for future resuming
        if not wandb_id:
            wandb_id_path = os.path.join(checkpoint_dir, 'wandb_id.txt')
            with open(wandb_id_path, 'w') as f:
                f.write(wandb.run.id)
    
    # Evaluate baseline (only if not resuming or no saved baseline)
    if baseline_metrics is None:
        print("Evaluating baseline on validation set...")
        baseline_metrics = evaluate_coherence(model, tokenizer, val_dataset)
        print(f"Baseline mean violation: {baseline_metrics['mean_violation']:.4f}")
        print(f"Baseline mean sum: {baseline_metrics['mean_sum']:.4f}")
        
        # Store baseline metrics
        all_eval_metrics.append(("Baseline", baseline_metrics))
        
        # Log baseline metrics to wandb
        if args.use_wandb:
            log_eval_metrics(baseline_metrics, epoch=0, improvement=0.0, wandb_enabled=True)
            wandb.log({"train_dataset_size": len(train_dataset), "val_dataset_size": len(val_dataset)})
    else:
        print(f"Using saved baseline - Mean violation: {baseline_metrics['mean_violation']:.4f}")
    
    # Training loop
    if start_epoch < config["num_epochs"]:
        print(f"Starting training from epoch {start_epoch + 1}...")
        model.train()
        
        # Get device from model
        device = next(model.parameters()).device
        
        for epoch in range(start_epoch, config["num_epochs"]):
            total_loss = 0
            num_batches = 0
            
            for batch_idx, batch in enumerate(train_dataloader):
                # Move to device using helper function
                batch = move_batch_to_device(batch, device)
                
                # Compute loss using selected loss function
                if config["loss_type"] == "smooth_xent":
                    loss = smooth_crossentropy_loss(
                        model, tokenizer, batch, 
                        use_softmax=config["use_softmax"], 
                        temperature=config["temperature"]
                    )
                else:  # default to MSE
                    loss = coherence_loss(model, tokenizer, batch)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                global_step += 1
                
                # Log to wandb
                if args.use_wandb:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/epoch": epoch + 1,
                        "train/step": global_step
                    })
                
                if num_batches % 10 == 0:
                    print(f"Epoch {epoch+1}, Batch {num_batches}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1} completed, Average Loss: {avg_loss:.4f}")
            
            # Evaluate after each epoch
            model.eval()
            epoch_metrics = evaluate_coherence(model, tokenizer, val_dataset)
            
            # Optionally also evaluate on training set for overfitting detection
            if args.eval_train:
                train_metrics = evaluate_coherence(model, tokenizer, train_dataset)
            
            model.train()
            
            # Store epoch metrics
            all_eval_metrics.append((f"Epoch {epoch + 1}", epoch_metrics))
            
            improvement = baseline_metrics['mean_violation'] - epoch_metrics['mean_violation']
            
            # Log epoch metrics to wandb
            if args.use_wandb:
                log_dict = {
                    "train/avg_loss": avg_loss,
                }
                if args.eval_train:
                    log_dict.update({
                        "train/mean_violation": train_metrics['mean_violation'],
                        "train/mean_sum": train_metrics['mean_sum'],
                        "overfitting/violation_gap": train_metrics['mean_violation'] - epoch_metrics['mean_violation'],
                        "overfitting/sum_gap": train_metrics['mean_sum'] - epoch_metrics['mean_sum']
                    })
                wandb.log(log_dict)
                
                # Log validation metrics with full visualizations
                log_eval_metrics(epoch_metrics, epoch=epoch + 1, improvement=improvement, wandb_enabled=True)
                
                # Also log training metrics with visualizations if eval_train is enabled
                if args.eval_train:
                    train_improvement = baseline_metrics['mean_violation'] - train_metrics['mean_violation']
                    log_eval_metrics(train_metrics, epoch=epoch + 1, improvement=train_improvement, wandb_enabled=True, prefix="train_eval")
            
            print(f"Epoch {epoch+1} validation - Mean violation: {epoch_metrics['mean_violation']:.4f}, "
                  f"Mean sum: {epoch_metrics['mean_sum']:.4f}, Improvement: {improvement:.4f}")
            
            if args.eval_train:
                train_improvement = baseline_metrics['mean_violation'] - train_metrics['mean_violation']
                violation_gap = train_metrics['mean_violation'] - epoch_metrics['mean_violation']
                print(f"Epoch {epoch+1} training   - Mean violation: {train_metrics['mean_violation']:.4f}, "
                      f"Mean sum: {train_metrics['mean_sum']:.4f}, Improvement: {train_improvement:.4f}")
                print(f"Overfitting gap: {violation_gap:.4f} (negative = overfitting)")
            
            # Save checkpoint
            if (epoch + 1) % args.save_every == 0 or (epoch + 1) == config["num_epochs"]:
                checkpoint_metrics = {
                    'baseline': baseline_metrics,
                    'all_eval_metrics': all_eval_metrics
                }
                checkpoint_path = save_checkpoint(
                    model, optimizer, epoch, global_step, config, checkpoint_metrics, checkpoint_dir
                )
                
                # Save wandb ID in checkpoint for resuming
                if args.use_wandb:
                    wandb_id_path = os.path.join(checkpoint_path, 'wandb_id.txt')
                    with open(wandb_id_path, 'w') as f:
                        f.write(wandb.run.id)
    
    ''' dont need this since we eval after every epoch
    # Final evaluation on validation set
    print("Evaluating final model on validation set...")
    final_metrics = evaluate_coherence(model, tokenizer, val_dataset)
    final_improvement = baseline_metrics['mean_violation'] - final_metrics['mean_violation']
    
    # Store final metrics
    if ("Final", final_metrics) not in all_eval_metrics:
        all_eval_metrics.append(("Final", final_metrics))
    
    print(f"Final mean violation: {final_metrics['mean_violation']:.4f}")
    print(f"Final mean sum: {final_metrics['mean_sum']:.4f}")
    print(f"Total improvement: {final_improvement:.4f}")
    '''
    
    # Log final metrics to wandb and create summary
    if args.use_wandb:
        # Log final point on timeline
        # log_eval_metrics(final_metrics, epoch=config["num_epochs"] + 1, improvement=final_improvement, wandb_enabled=True)
        
        # Create comprehensive summary table from stored metrics
        summary_data = []
        for stage, metrics in all_eval_metrics:
            improvement = baseline_metrics['mean_violation'] - metrics['mean_violation'] if stage != "Baseline" else 0.0
            summary_data.append([
                stage,
                f"{metrics['mean_violation']:.4f}",
                f"{metrics['max_violation']:.4f}",
                f"{metrics['mean_sum']:.4f}",
                f"{improvement:.4f}"
            ])
        
        summary_table = wandb.Table(
            columns=["Stage", "Mean Violation", "Max Violation", "Mean Sum", "Improvement"],
            data=summary_data
        )
        wandb.log({"training_summary": summary_table})
        
        wandb.finish()


if __name__ == "__main__":
    main()