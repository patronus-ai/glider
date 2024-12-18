import json
import os
import argparse
import torch
from vllm import LLM, SamplingParams
from datasets import load_dataset
from huggingface_hub import get_collection

NUM_DEVICES = torch.cuda.device_count()

def run_test(d, json_filepath="results.jsonl"):
    inputs = []
    for prompt in d["prompt"]:
        inputs.append([{"role": "user", "content": prompt}])
    outputs = llm.chat(inputs, sampling_params)
    
    results = []
    for output, elem in zip(outputs, d):
        generated_text = output.outputs[0].text
        result = {"prompt": elem["prompt"], "ground_truth": elem["score"], "generated": generated_text, **elem}
        results.append(result)
    
    with open(json_filepath, 'w') as f:
        f.write('\n'.join(map(json.dumps, results)))

def run_tests(output_dir):
    collection = get_collection("PatronusAI/slm-evaluator-suite-673ceadb04c69353dd2a13fe")
    for dataset in collection.items:
        data = load_dataset(dataset.item_id)
        for i in data:
            split = data[i]
            print(f"Running {dataset.item_id} {'' if i in ['train', 'test'] else f'_{i}'} set evaluation\n")
            run_test(split, f"{output_dir}/{dataset.item_id.split('/')[-1]}{'' if i in ['train', 'test'] else '_{}'.format(i)}.jsonl")
    
    print("All evaluations completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="model/", help="Path to model directory or HF path")
    parser.add_argument("--output_dir", type=str, default="eval_outputs/", help="Path to output directory")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=2048)
    llm = LLM(model=args.model, 
              tensor_parallel_size=NUM_DEVICES, 
              dtype="bfloat16", 
              max_seq_len_to_capture=131072,
              # max_model_len=8192, # Uncomment this if running into OOM issues
             )

    run_tests(args.output_dir)