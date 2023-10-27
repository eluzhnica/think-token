import transformers
import torch

class CustomTrainer(transformers.Trainer):
    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = tokenizer

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            if self.args.past_index >= 0 and inputs["input_ids"].shape[1] > 1:
                inputs["inputs_embeds"] = model.get_input_embeddings()(
                    inputs["input_ids"][:, :-1]
                )
                inputs["use_cache"] = True
                inputs["past_key_values"] = inputs["past_key_values"][:, :-1]

            # Get the prompt from the inputs
            prompts = self.tokenizer.batch_decode(inputs["input_ids"], skip_special_tokens=True)
            completions = [prompt.split("---")[1] for prompt in prompts]
            prompts = [prompt.split("---")[0] + "---" for prompt in prompts]
            
            # Set eos to pad before doing the encoding
            self.tokenizer.padding_side = "left"
            prompts = self.tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True, truncation=True)["input_ids"].to(self.args.device)
            completions = self.tokenizer.batch_encode_plus(completions, return_tensors="pt", padding=True, truncation=True)["input_ids"].to(self.args.device)
            self.tokenizer.padding_side = "right"

            # Generate the output sequence
            actual_model = getattr(model, "module", model)
            generated_tokens = actual_model.generate(
                input_ids=prompts,
                do_sample=False,
                min_length=1,
                max_length=inputs["input_ids"].shape[1],
                pad_token_id=actual_model.config.pad_token_id,
                eos_token_id=actual_model.config.eos_token_id,
            )

            preds = generated_tokens
            loss = None

        return (loss, preds, completions)
    