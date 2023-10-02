# Generates positive movie reviews by tuning a pretrained model on IMDB dataset
# with a sentiment reward function
import json
import os
import sys
from typing import List

from datasets import load_dataset
from transformers import DistilBertForSequenceClassification, pipeline

import trlx
from trlx.data.default_configs import (
    TRLConfig,
    default_nemo_1_3b_config,
    default_nemo_2b_config,
    default_nemo_20b_config,
    default_ppo_config,
)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


def get_positive_score(scores):
    "Extract value associated with a positive sentiment from pipeline's output"
    return dict(map(lambda x: tuple(x.values()), scores))["POSITIVE"]


def main(hparams={}):
    # Merge sweep config with default config if given
    default_config = TRLConfig.update(default_ppo_config().to_dict(), hparams)

    cfg_name = os.environ.get("NEMO_CONFIG", "1.3B")
    if cfg_name == "1.3B":
        nemo_config = default_nemo_1_3b_config()
    elif cfg_name == "2B":
        nemo_config = default_nemo_2b_config()
    elif cfg_name == "20B":
        nemo_config = default_nemo_20b_config()
    else:
        raise ValueError(f"Unknown NEMO_CONFIG: {cfg_name}")

    config = default_config.evolve(
        train=dict(
            total_steps=512,
            seq_length=2048,
            batch_size=32,
            epochs=100,
            eval_interval=64,
            trainer="NeMoPPOTrainer",
            trainer_kwargs=dict(
                pretrained_model=f"/mnt/hdd/nemo-megatron-gpt-{cfg_name}/",
                megatron_cfg=nemo_config,
            ),
            checkpoint_interval=256,
            checkpoint_dir=f"nemo_{cfg_name}_ppo_sentiments",
            seed=2023,
            project_name="trlxnemo",
            tags=["nemo", "ppo", "sentiments", cfg_name],
        ),
        optimizer=dict(
            name="distributed_fused_adam",
            kwargs=dict(
                lr=6.001e-5,
                weight_decay=1e-06,
                eps=1.0e-8,
                betas=(0.9, 0.95),
            ),
        ),
        scheduler=dict(
            name="CosineAnnealing",
        ),
        model=dict(num_layers_unfrozen=2),
        method=dict(
            num_rollouts=128,
            init_kl_coef=0.05,
            scale_reward="ref",
            vf_coef=1,
            gen_kwargs=dict(temperature=1.0, max_new_tokens=40),
            chunk_size=128,
            ppo_epochs=4,
        ),
    )
    config.scheduler.kwargs = dict(warmup_steps=0, constant_steps=1e12, min_lr=6.0e-5)

    rank = int(os.environ["SLURM_PROCID"])
    local_rank = rank % 8
    
    checkpoint_path = '/home/asap7772/reward_model_test/model_checkpoints_rewpref/distilgpt2_imdb_pospref_sft-review-model-20230926-232443_gpt2-xl_20230927-224048_wordcutoff_0_10_1_1e-05/20230929-204118'
    reward_model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)
    reward_tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    reward_tokenizer.add_special_tokens({"pad_token": "<|padding|>"})
    reward_tokenizer.padding_side = "right"
    reward_tokenizer.truncation_side = "left"
    reward_model.to(local_rank)
    reward_model.eval()

    def reward_fn(samples: List[str], **kwargs) -> List[float]:
        # Add eos token to each text
        for i, te in enumerate(samples):
            if not te.endswith(reward_tokenizer.eos_token):
                samples[i] = te + reward_tokenizer.eos_token

        with torch.no_grad():
            encoded_input = reward_tokenizer(samples, return_tensors='pt')
            encoded_input.to(local_rank)
            output = reward_model(**encoded_input)
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
