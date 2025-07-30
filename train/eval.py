import torch
import math
from transformers import TrainerCallback
from torch.utils.data import DataLoader


class PerplexityCallback(TrainerCallback):
    def __init__(self, eval_dataset, data_collator, tokenizer, batch_size=8):
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def on_evaluate(self, args, state, control, model=None, eval_dataloader=None, **kwargs):
        if model is not None and self.eval_dataset is not None:
            perplexity = evaluate_by_perplexity(
                model=model,
                eval_dataset=self.eval_dataset,
                data_collator=self.data_collator,
                tokenizer=self.tokenizer,
                batch_size=self.batch_size
            )

            print(f"Step {state.global_step}: Perplexity = {perplexity:.4f}")

            if hasattr(state, 'log_history') and len(state.log_history) > 0:
                state.log_history[-1]['eval_perplexity'] = perplexity


def evaluate_by_perplexity(model, eval_dataset, data_collator, tokenizer, batch_size=8, device=None):
    """
    Calculate perplexity on the evaluation dataset

    Args:
        model: The model to evaluate
        eval_dataset: Dataset to evaluate on
        data_collator: Data collator for batching
        tokenizer: Tokenizer
        batch_size: Batch size for evaluation
        device: Device to run evaluation on

    Returns:
        float: Perplexity score
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    total_loss = 0.0
    total_tokens = 0

    dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        collate_fn=data_collator,
        shuffle=False
    )

    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss

            valid_tokens = (batch['labels'] != -100).sum().item()

            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')

    model.train()
    return perplexity
