import json
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer
from datasets import load_from_disk, DatasetDict
import torch
from dataclasses import dataclass
from typing import Any, Dict, List

from train.eval import PerplexityCallback, evaluate_by_perplexity


@dataclass
class DataCollatorForCompletionLM:
    tokenizer: Any
    pad_to_multiple_of: int = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        input_ids = [feature["input_ids"] for feature in features]
        labels = [feature["labels"] for feature in features]

        max_length = max(len(ids) for ids in input_ids)

        padded_input_ids = []
        padded_labels = []
        attention_mask = []

        for ids, lbls in zip(input_ids, labels):
            padding_length = max_length - len(ids)

            padded_ids = ids + [self.tokenizer.pad_token_id] * padding_length
            mask = [1] * len(ids) + [0] * padding_length

            padded_lbls = lbls + [-100] * padding_length

            padded_input_ids.append(padded_ids)
            padded_labels.append(padded_lbls)
            attention_mask.append(mask)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long)
        }


def preprocess_function(examples, tokenizer, max_length=1024, window_size=None, window_step=None):
    if window_size is None or window_step is None:
        texts = [text + tokenizer.eos_token for text in examples["text"]]
        tokenized = tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=max_length,
            return_overflowing_tokens=False,
        )
        tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
        return tokenized

    else:
        text_samples = examples["text"]
        windowed_texts = []

        for start_idx in range(0, len(text_samples), window_step):
            end_idx = min(start_idx + window_size, len(text_samples))

            if end_idx - start_idx < window_size and start_idx > 0:
                break

            window_texts = text_samples[start_idx:end_idx]
            combined_text = tokenizer.eos_token.join(window_texts) + tokenizer.eos_token
            windowed_texts.append(combined_text)

        if not windowed_texts and text_samples:
            combined_text = tokenizer.eos_token.join(text_samples) + tokenizer.eos_token
            windowed_texts.append(combined_text)

        tokenized = tokenizer(
            windowed_texts,
            truncation=True,
            padding=False,
            max_length=max_length,
            return_overflowing_tokens=False,
        )

        tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]]
        return tokenized


def train():
    model_name = "Qwen/Qwen3-8B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    model.resize_token_embeddings(len(tokenizer))

    dataset = load_from_disk("/sc/home/valentin.teutschbein/pack/datasets/completion_dataset/")

    max_length = 1024
    tokenized_dataset = DatasetDict()

    for split in dataset.keys():
        tokenized_dataset[split] = dataset[split].map(
            lambda examples: preprocess_function(examples, tokenizer, max_length, window_size=8, window_step=4),
            batched=True,
            remove_columns=dataset[split].column_names,
        )

    data_collator = DataCollatorForCompletionLM(tokenizer=tokenizer)

    perplexity_callback = PerplexityCallback(
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        batch_size=4
    )

    test_perplexity = evaluate_by_perplexity(
        model=model,
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        batch_size=4
    )
    print(f"Initial test perplexity: {test_perplexity:.4f}")

    training_args = TrainingArguments(
        output_dir="./qwen-finetuned",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        learning_rate=5e-5,
        logging_steps=10,
        logging_dir="./logs",
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=10,
        save_steps=100,
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,
        callbacks=[perplexity_callback],
    )

    trainer.train()
    trainer.save_model()
    tokenizer.save_pretrained("./qwen-finetuned")

    test_perplexity = evaluate_by_perplexity(
        model=model,
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        batch_size=4
    )

    test_results = trainer.evaluate(tokenized_dataset["test"])
    test_results['test_perplexity'] = test_perplexity

    print(f"Test results: {test_results}")
    print(f"Final test perplexity: {test_perplexity:.4f}")

    with open("./qwen-finetuned/training_logs.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=4)


if __name__ == "__main__":
    train()
