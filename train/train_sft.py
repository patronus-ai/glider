import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer


def load_dataset(dataset_path):
    train = load_dataset(dataset_path, split="train")
    eval = load_dataset(dataset_path, split="eval")

    def train_mapper(x):
        return {"messages": [{"role": "user", "content": x["prompt"]}, {"role": "assistant", "content": x["chosen"]}]}
    
    train = train.map(train_mapper, batched=False, remove_columns=[i for i in train.column_names if "messages" not in i], num_proc=15)
    eval = eval.map(train_mapper, batched=False, remove_columns=[i for i in eval.column_names if "messages" not in i], num_proc=15)

    return train, eval


def train_model(train, eval, model_dir, model_output_path):
    model = AutoModelForCausalLM.from_pretrained(model_dir, 
                                                attn_implementation="flash_attention_2", 
                                                torch_dtype=torch.bfloat16,
                                                use_cache=False,
                                                cache_dir="model/", low_cpu_mem_usage=True)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.padding_side = 'right'

    sft_config = SFTConfig(
        max_seq_length=8192,
        output_dir=model_output_path,
        learning_rate=5e-5,
        num_train_epochs=1,
        gradient_checkpointing=True,
        save_strategy="epoch",
        eval_strategy="epoch",
        dataset_batch_size=20000,
        bf16=True,
        load_best_model_at_end=True,
        gradient_accumulation_steps=4,
        eval_accumulation_steps=4,
        save_total_limit = 5,
        logging_dir=model_output_path,
        logging_strategy='steps',
        logging_steps=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        lr_scheduler_type= "cosine",
        warmup_ratio= 0.03,
        weight_decay= 0.01,
        report_to="wandb",
        run_name="train_sft_phi",
    )

    trainer = SFTTrainer(
        model,
        train_dataset=train,
        eval_dataset=eval,
        args=sft_config,
        tokenizer=tokenizer,
    )

    trainer.train(resume_from_checkpoint=False)
    trainer.save_model(model_output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="microsoft/phi-3.5-mini", help="Path to model directory or HF path")
    parser.add_argument("--model_output_path", type=str, default="phi_outputs/", help="Path to output directory")
    parser.add_argument("--dataset_path", type=str, default="", help="Path to dataset")

    args = parser.parse_args()
    train, eval = load_dataset(args.dataset_path)
    train_model(train, eval, args.model_dir, args.model_output_path)