import torch
import transformers
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, EvalPrediction
from datasets import load_dataset, Dataset
import os
import argparse
import wandb
from datacollators.forward_looking import DataCollatorForLanguageModelingForwardLooking

from tokenizer import construct_tokenizer
from trainers.generation_eval_trainer import CustomTrainer
import random
import numpy as np

if os.getenv('WANDB_MODE') is None:
    wandb.init(mode="disabled")


# Set seeds for reproducibility
# torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


def collatz(n):
    if n == 1:
        return 1
    elif n % 2 == 0:
        return 1 + collatz(n // 2)
    else:
        return 1+ collatz(3*n + 1)

def multiply(a, b):
    return a * b

def construct_multiply_dataset(datapoints: int = 1000):
    dataset = {}
    while len(dataset) < datapoints:
        a = random.randint(100, 999)
        b = random.randint(100, 999)
        dataset[(a, b)] = a * b

    return dataset


def parse_output(text):
    try:
        number_str = text.split('---')[1]
        number_str = number_str.replace("[compute]", "")
        number = int(number_str.strip())
    except Exception as e:
        number = 0
    return number

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = logits.argmax(-1)
    return pred_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compute_tokens', type=int, default=5)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--n_embed', type=int, default=128)
    args = parser.parse_args()

    tokenizer = construct_tokenizer()

    config = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)
    config.n_head = 8
    config.n_layer = args.depth
    config.n_embd = args.n_embed
    config.n_positions = 1024
    config.n_ctx = 1024
    config.pad_token_id = tokenizer.pad_token_id
    model = GPT2LMHeadModel(config)
    
    # tokenizer = AutoTokenizer.from_pretrained("gpt2")

    # new_token = "[compute]"
    # tokenizer.add_tokens([new_token])
    
    model.resize_token_embeddings(len(tokenizer))

    params = sum(p.numel() for p in model.parameters())
    params_in_billion = params / (10**9)
    print(f"Parameters: {params_in_billion} B")
    # print in thousands too
    print(f"Parameters: {params_in_billion * 1000} K")

    def compute_accuracy(p: EvalPrediction):
        labels = p.label_ids
        preds = p.predictions
        inputs = p.inputs
        # print(preds)
        # pred_ids = preds.argmax(-1)
        pred_ids = preds
        
        pred_ids = [[pred for pred in pred_array if pred != -100] for pred_array in pred_ids]
        pred_texts = [tokenizer.decode(pred, skip_special_tokens=True) for pred in pred_ids]
        # print(pred_texts)
        filtered_labels = [[label for label in label_array if label != -100] for label_array in labels]
        label_numbers = [int(tokenizer.decode(label, skip_special_tokens=True).strip()) for label in filtered_labels]
        pred_numbers = [parse_output(text) for text in pred_texts]
        # print(list(zip(pred_numbers, pred_texts)))
        # print(label_numbers)
        assert len(pred_numbers) == len(label_numbers), "Length of predictions and labels should be equal"        
        accuracy = sum(pred == label for pred, label in zip(pred_numbers, label_numbers)) / len(pred_numbers)
        # Log into a wandb table the input, predictions, and true labels

        if trainer.state.global_step >= trainer.state.max_steps - 1 or trainer.state.epoch >= trainer.state.num_train_epochs - 1:
            filtered_inputs = [[inp for inp in inp_array if inp != -100] for inp_array in inputs]
            input_texts = [tokenizer.decode(inp, skip_special_tokens=True) for inp in filtered_inputs]
            wandb_table = wandb.Table(columns=["Input", "Prediction text", "Predictions", "True Labels"])
            for input_text, pred_text, pred, label in zip(input_texts, pred_texts, pred_numbers, label_numbers):
                wandb_table.add_data(input_text, pred_text, pred, label)
            wandb.log({"Prediction Table": wandb_table})

        # Calculate MSE loss
        mse_loss = torch.nn.functional.mse_loss(torch.tensor(pred_numbers).float(), torch.tensor(label_numbers).float())

        return {'schaccuracy': accuracy, "mse_loss": mse_loss}

    # collatz_dict = {n: collatz(n) for n in range(1, 100001)}
    multiply_dict = construct_multiply_dataset(200000) 
    dataset = list(map(lambda item: f"{item[0][0]}*{item[0][1]}{'[compute]'*args.compute_tokens}---{item[1]}{tokenizer.eos_token}", multiply_dict.items()))
    dataset = Dataset.from_dict({"text": dataset})
    dataset = dataset.map(lambda samples: tokenizer(samples["text"]), batched=True)
    # dataset = dataset.map(lambda samples: process_dataset(tokenizer, samples), batched=True)
    # remove "text" column
    dataset = dataset.remove_columns(["text"])
    data = dataset

    data = data.train_test_split(test_size=0.2, seed=42)
    train_data, test_data = data["train"], data["test"]
    train_data = train_data.train_test_split(test_size=0.1, seed=42)
    train_data, dev_data = train_data["train"], train_data["test"]
    data = {"train": train_data, "dev": dev_data, "test": test_data}

    # tokenizer.pad_token = tokenizer.eos_token
    os.environ["WANDB_PROJECT"] = "think-hard"
    run_name_modifier = "-constant-paramm" if args.n_embed != 128 else ""
    trainer = CustomTrainer(
        tokenizer=tokenizer,
        model=model,
        train_dataset=data["train"],
        eval_dataset=data["dev"],
        args=transformers.TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.accumulation_steps,
            warmup_steps=20,
            logging_strategy="steps",
            num_train_epochs=args.epochs,
            learning_rate=2e-3,
            logging_steps=100,
            evaluation_strategy="epoch",
            per_device_eval_batch_size=args.batch_size,
            output_dir="outputs",
            report_to="wandb",
            run_name=f"gpts-digit-multiply-fixed-dataset-{args.compute_tokens}-ic-deep-{config.n_layer}" + run_name_modifier,
            include_inputs_for_metrics=True,
        ),
        data_collator=DataCollatorForLanguageModelingForwardLooking(tokenizer, mlm=False),
        compute_metrics=compute_accuracy,
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics 
    )
    trainer.train()
    wandb.config.compute_tokens = args.compute_tokens
    wandb.config.n_layer = config.n_layer
    wandb.run.log_code(".")

if __name__ == "__main__":
    main()
