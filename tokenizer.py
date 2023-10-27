

def construct_tokenizer(file_path: str="tokenizer.json"):
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
    from transformers import GPT2TokenizerFast

    import os

    special_tokens = {"pad_token": "[PAD]", "unk_token": "[UNK]", "cls_token": "[CLS]", "sep_token": "[SEP]", "mask_token": "[MASK]", "eos_token": "[SEP]", "bos_token": "[CLS]"}

    # Check if tokenizer.json exists
    if os.path.exists(file_path):
        # Load PreTrainedTokenizer from tokenizer.json
        tokenizer = GPT2TokenizerFast(tokenizer_file=file_path)
        tokenizer.add_special_tokens(special_tokens)
    else:
        # Define the vocabulary
        # Define the special tokens
        # Define the additional tokens
        additional_tokens = ["+", "-", "*", " ", "---"] + list("0123456789")

        # Initialize a tokenizer
        tokenizer = Tokenizer(models.BPE())

        # Customize pre-tokenization and decoding
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.decoder = decoders.BPEDecoder()

        # Add special tokens to the tokenizer
        tokenizer.add_special_tokens(list(special_tokens.values()))
        
        # Add additional tokens to the tokenizer
        tokenizer.add_tokens(additional_tokens)
        tokenizer.add_tokens(["[compute]"])

        # Post-processing: Truncation & Padding
        tokenizer.enable_truncation(max_length=1024)

        # Save the tokenizer to tokenizer.json
        tokenizer.save(file_path)
        tokenizer = GPT2TokenizerFast(tokenizer_file=file_path)
    tokenizer.model_max_length = 1024
    return tokenizer
    # tokenizer.enable_padding(length=512, pad_id=tokenizer.token_to_id("[PAD]"))
