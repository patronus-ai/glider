# Patronus GLIDER
<div align="center">

[![arXiv](https://img.shields.io/badge/cs.LG-arXiv%3A2412.14140-b31b1b)](https://arxiv.org/abs/2412.14140)
[![HuggingFace Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/PatronusAI/GLIDER)
	[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21E?logo=huggingface&logoColor=000)](https://huggingface.co/PatronusAI/glider)
</div>

This repository contains the evaluation code and test sets for Patronus' GLIDER model. 

### How to run training:

The training is done in two phases: SFT and alignment using RLAIF data. To train your model, first install the conda environment in the `train_environment.yaml` file:

```bash
conda env create -f train_environment.yaml
```

If you wish to install flash attention (which is enabled by default), you must additionally activate enviornment and run the following

```bash
pip install flash-attn
```

Once the environment is set up properly, you need to setup HF accelerate configs according to your GPU configuration. To do this, you can run:

```bash
accelerate config
```

The GLIDER model was trained with FSDP which can be enabled using the config setup above. 

Finally, to run the training:

```bash
cd train
python train_sft.py --model_output_path="[your_output_dir_here]" --dataset_path="[your dataset path here]"
python train_dpo.py --model_dir="[your_sft_saved_model_path]" --model_output_path="[your_output_dir_here]" --dataset_path="[your preference dataset path here]"
```

Once the training is done, you can evaluate the model using the script below. Ensure that you have `vllm` and `scikit-learn` installed:

```bash
cd eval
python vllm_test_local.py --model="[your_model_path]" --output_dir="[dir_to_save_your_outputs]"
python extract_accuracies.py --eval_dir="[your_outputs_path_above]" --output_file="[file_to_write_results_to.jsonl]"
```

Note that vLLM has an issue with the longrope implementation of phi models which is being actively fixed at the time of release of this repo. It is recommended to use [this](https://github.com/garg-amit/vllm/tree/fix-long-seq-bug) version of vLLM till the fix PR is merged into vLLM's main branch.

This code is licensed under CC-BY-NC-4.0. More information is available in the `LICENSE` file.
