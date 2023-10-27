import torch
import transformers
from transformers import AutoTokenizer, GPT2Config, EvalPrediction, GPT2LMHeadModel
from datasets import load_dataset, Dataset
import os
import argparse
import wandb
import numpy as np

from tokenizer import construct_tokenizer
from trainers.generation_eval_trainer import CustomTrainer


if os.getenv('WANDB_MODE') is None:
    wandb.init(mode="disabled")


def safe_convert_to_tensor(numbers):
    max_float32 = np.finfo(np.float32).max
    min_float32 = np.finfo(np.float32).min
    capped_numbers = []
    for number in numbers:
        if isinstance(number, (int, float)):
            if number > max_float32:
                capped_numbers.append(float(max_float32))
            elif number < min_float32:
                capped_numbers.append(float(min_float32))
            else:
                capped_numbers.append(float(number))
        else:
            capped_numbers.append(0.0)
    return torch.tensor(capped_numbers).float()


def collatz(n):
    sequence = [n]
    while n != 1:
        if n % 2 == 0:
            n = n // 2
        else:
            n = 3*n + 1
        sequence.append(n)
    # sequence.append(len(sequence))
    return sequence


def parse_output(text):
    try:
        number_strs = text.split('---')[1].split('[compute]')
        numbers = [int(number_str.strip()) if number_str.strip().isdigit() else 0 for number_str in number_strs]
    except Exception as e:
        numbers = [0]
    return numbers


def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = logits.argmax(-1)
    return pred_ids

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--compute_tokens', type=int, default=950)
    parser.add_argument('--accumulation_steps', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=2)
    args = parser.parse_args()

    tokenizer = construct_tokenizer()
    config = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)
    config.n_head = 4
    config.n_layer = 6
    config.n_embd = 128
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
        pred_ids = preds

        pred_texts = [tokenizer.decode([x for x in pred if x != -100], skip_special_tokens=True) for pred in pred_ids]
        filtered_labels = [[label for label in label_array if label != -100] for label_array in labels]
        label_numbers = [[int(x) for x in tokenizer.decode(label, skip_special_tokens=True).split("[compute]")] for label in filtered_labels]
        pred_numbers = [parse_output(text) for text in pred_texts]

        assert len(pred_numbers) == len(label_numbers), "Length of predictions and labels should be equal"

        # Calculate accuracy and MSE loss for the last output
        # Find the position of the first "1" in the network outputs
        first_one_positions = [pred.index(1) if 1 in pred else -1 for pred in pred_numbers]
        first_one_labels = [label.index(1) if 1 in label else -1 for label in label_numbers] # this shouldn't happen but just in case
        # Compare it with the length of the label_numbers
        label_lengths = [len(label) for label in label_numbers]

        # Calculate accuracy and MSE loss based on the position of the first "1"
        first_one_accuracy = sum(pred == label for pred, label in zip(first_one_positions, label_lengths)) / len(first_one_positions)
        first_one_mse_loss = torch.nn.functional.mse_loss(torch.tensor(first_one_positions).float(), torch.tensor(label_lengths).float())

        # Log into a wandb table the input, predictions, and true labels
        filtered_inputs = [[inp for inp in inp_array if inp != -100] for inp_array in inputs]
        input_texts = [tokenizer.decode(inp, skip_special_tokens=True) for inp in filtered_inputs]
        wandb_table = wandb.Table(columns=["Input", "Prediction text", "Collatz Steps Prediction", "Collatz Steps True", "Predictions", "True Labels", ])
        for input_text, pred_text, pred, label, first_one, true_one in zip(input_texts, pred_texts, pred_numbers, label_numbers, first_one_positions, label_lengths):
            wandb_table.add_data(input_text, pred_text, first_one, true_one, pred, label)
        wandb.log({"Prediction Table": wandb_table})
        
        # max_length = max(max(len(pred) for pred in pred_numbers), max(len(label) for label in label_numbers))
        # accuracy_per_position = []
        # mse_loss_per_position = []
        # for i in range(max_length):
        #     pred_i = [pred[i] if i < len(pred) else None for pred in pred_numbers]
        #     label_i = [label[i] if i < len(label) else None for label in label_numbers]
        #     valid_indices = [index for index, (pred, label) in enumerate(zip(pred_i, label_i)) if pred is not None and label is not None]
        #     if valid_indices:
        #         accuracy_i = sum(pred_i[index] == label_i[index] for index in valid_indices) / len(valid_indices)
        #         mse_loss_i = torch.nn.functional.mse_loss(safe_convert_to_tensor([pred_i[index] for index in valid_indices]), safe_convert_to_tensor([label_i[index] for index in valid_indices]))
        #         accuracy_per_position.append(accuracy_i)
        #         mse_loss_per_position.append(mse_loss_i)

        # # Convert lists to dictionaries for logging
        # #accuracy_per_position_dict = {f'accuracy_at_pos_{i}': acc for i, acc in enumerate(accuracy_per_position)}
        # #mse_loss_per_position_dict = {f'mse_loss_at_pos_{i}': loss for i, loss in enumerate(mse_loss_per_position)}

        # # Calculate average accuracy and MSE loss across all positions
        # avg_accuracy_per_position = np.array(accuracy_per_position).mean()
        # avg_mse_loss_per_position = np.array(mse_loss_per_position).mean()

        return {'schaccuracy': first_one_accuracy, "mse_loss": first_one_mse_loss} #"avg_accuracy_per_position": avg_accuracy_per_position, "avg_mse_loss_per_position": avg_mse_loss_per_position} #**accuracy_per_position_dict, **mse_loss_per_position_dict}

    #collatz_dict = {n: collatz(n) for n in range(1, 100001)}
    #dataset = list(map(lambda item: f"{item[0]} \n --- {'[compute]'*args.compute_tokens} {item[1]} {tokenizer.eos_token}", collatz_dict.items()))
    collatz_dict = {n: collatz(n) for n in range(1, 100001)}
    dataset = list(map(lambda item: f"{item[0]}---{'[compute]'.join(map(str, item[1]))}{tokenizer.eos_token}", collatz_dict.items()))
    dataset = Dataset.from_dict({"text": dataset})
    dataset = dataset.map(lambda samples: tokenizer(samples["text"]), batched=True)
    dataset = dataset.filter(lambda x: len(x['input_ids']) <= 1000)  # Filter out examples longer than 1000 tokens
    data = dataset

    data = data.train_test_split(test_size=0.2, seed=42)
    train_data, test_data = data["train"], data["test"]
    train_data = train_data.train_test_split(test_size=0.1, seed=42)
    train_data, dev_data = train_data["train"], train_data["test"]
    data = {"train": train_data, "dev": dev_data, "test": test_data}

    tokenizer.pad_token = tokenizer.eos_token
    os.environ["WANDB_PROJECT"] = "think-hard"
    
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
            num_train_epochs=250,
            learning_rate=1e-3,
            logging_steps=100,
            evaluation_strategy="epoch",
            #eval_steps=200,
            per_device_eval_batch_size=args.batch_size,
            # eval_accumulation_steps=args.batch_size,
            output_dir="outputs",
            report_to="wandb",
            run_name=f"gpts-digit-collatz-sequence-ic",
            include_inputs_for_metrics=True,
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        compute_metrics=compute_accuracy,
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics 
    )
    trainer.train()
    wandb.config.n_layer = config.n_layer
    wandb.run.log_code(".")


if __name__ == "__main__":
    main() 
