# Generates positive movie reviews by tuning a pretrained model on IMDB dataset
# with a sentiment reward function
import json
import os
import sys
from typing import List

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datasets import load_dataset
from transformers import pipeline

import trlx
from trlx.data.default_configs import TRLConfig, default_ppo_config


def main(hparams={}):
    # Merge sweep config with default config if given
    checkpoint_path = '/home/asap7772/reward_model_test/model_checkpoints_rewpref/distilgpt2_imdb_pospref_sft-review-model-20230926-232443_gpt2-xl_20230927-224048_wordcutoff_0_10_1_1e-05/20230929-204118'
    hparams['reward_checkpoint_path'] = checkpoint_path
    config = TRLConfig.update(default_ppo_config().to_dict(), hparams)

    if torch.cuda.is_available():
        device = int(os.environ.get("LOCAL_RANK", 0))
    else:
        device = -1

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    tokenizer.add_special_tokens({"pad_token": "<|padding|>"})
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "left"
    model.to(device)
    model.eval()

    def reward_fn(samples: List[str], **kwargs) -> List[float]:
        # Add eos token to each text
        for i, te in enumerate(samples):
            if not te.endswith(tokenizer.eos_token):
                samples[i] = te + tokenizer.eos_token

        with torch.no_grad():
            encoded_input = tokenizer(samples, return_tensors='pt', padding=True, truncation=True)
            encoded_input.to(device)
            output = model(**encoded_input)
            logits = output.logits.squeeze().cpu().numpy()
        return logits.tolist()

    # Take few words off of movies reviews as prompts
    imdb = load_dataset("imdb", split="train+test")
    prompts = [" ".join(review.split()[:4]) for review in imdb["text"]]
    
    rot_tom = load_dataset("rotten_tomatoes", split="train+test")
    eval_prompts = [" ".join(review.split()[:4]) for review in rot_tom["text"]][:256]

    trlx.train(
        reward_fn=reward_fn,
        prompts=prompts,
        eval_prompts=eval_prompts,
        config=config,
    )


if __name__ == "__main__":
    hparams = {} if len(sys.argv) == 1 else json.loads(sys.argv[1])
    main(hparams)
