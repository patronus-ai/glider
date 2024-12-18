import datetime
import time
import os
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from trl import DPOConfig, DPOTrainer
from transformers import EarlyStoppingCallback


def load_data(dataset_path):
    train = load_dataset(dataset_path, split="train")
    eval = load_dataset(dataset_path, split="eval")
    return train, eval

def train_model(train, eval, model_dir, model_output_path):
    base_model = AutoModelForCausalLM.from_pretrained(model_dir, 
                                                attn_implementation="flash_attention_2", 
                                                torch_dtype=torch.bfloat16,
                                                use_cache=False,
                                                cache_dir="model/",
                                                device_map=f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)


    training_args = DPOConfig(
        beta=0.1,
        learning_rate=5e-7,
        per_device_train_batch_size = 2,
        per_device_eval_batch_size = 2,
        num_train_epochs=1,
        gradient_checkpointing = False,
        warmup_ratio = 0.05,
        lr_scheduler_type = "cosine",
        max_grad_norm = 1.0,
        load_best_model_at_end=True,
        output_dir=model_output_path,
        save_strategy='steps',
        save_steps = 500,
        save_total_limit = 2,
        logging_dir=model_output_path,
        logging_strategy = 'steps',
        logging_steps = 1,
        eval_strategy = 'steps',
        eval_steps = 500,
        bf16=True,
        max_length=8192,
        max_prompt_length=8000,
        dataloader_num_workers=32,
        dataloader_prefetch_factor=32,
        report_to="wandb",
        run_name=f"eval_{datetime.datetime.today().strftime('%m/%d/%Y')}_{time.time()}",
        dataset_num_proc=30,
        loss_type="apo_zero",
        rpo_alpha=1,
    )

    dpo_trainer = DPOTrainer(
        base_model,
        ref_model=None,
        args=training_args,
        train_dataset=train,
        eval_dataset=eval,
        tokenizer=tokenizer,
    )

    dpo_trainer.train(resume_from_checkpoint = False)
    dpo_trainer.save_model(model_output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="phi_outputs/",
        help="Path to the SFT trained model",
    )
    parser.add_argument(
        "--model_output_path",
        type=str,
        default="apo_zero_phi_v9",
        help="Path to the output model",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="dataset_path",
        help="Path to the dataset (local or HF)",
    )
    args = parser.parse_args()

    train, eval = load_data(args.dataset_path)
    train_model(train, eval, args.model_path, args.model_output_path)
