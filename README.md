# ALTeGraD Labs

Lab notebooks for **APM 53674: ALTeGraD** (Advanced Learning for Text and Graph Data), taught by Prof. Michalis Vazirgiannis. The labs were run by Yang Zhang and Amr Mohamed. They cover sequence-to-sequence models, then LLM fine-tuning and post-training.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Transformers%20%7C%20PEFT%20%7C%20TRL-yellow)
![Colab](https://img.shields.io/badge/Google%20Colab-T4%20GPU-F9AB00?logo=googlecolab&logoColor=white)

## Contents

| # | Lab | Topics | Open |
|---|-----|--------|------|
| 1 | [Neural Machine Translation and Language Modeling](Neural_Machine_Translation_and_Language_Modeling.ipynb) | Seq2seq, GRU encoder/decoder, global attention (Luong et al., 2015), training and evaluation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jrejorio/altegrad/blob/main/Neural_Machine_Translation_and_Language_Modeling.ipynb) |
| 2 | [Pretraining and Supervised Finetuning](Pretraining_and_supervised_finetuning.ipynb) | 4-bit quantization (bitsandbytes), LoRA/PEFT, chat templates, summarization SFT, tool-calling SFT | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jrejorio/altegrad/blob/main/Pretraining_and_supervised_finetuning.ipynb) |
| 3 | [Post-Training LLMs with DPO and GRPO](Post_Training_LLMs_with_DPO_and_GRPO.ipynb) | Off-policy DPO, on-policy GRPO, reward design, W&B tracking, GSM8K evaluation | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jrejorio/altegrad/blob/main/Post_Training_LLMs_with_DPO_and_GRPO.ipynb) |

## Lab summaries

### Lab 1: Neural Machine Translation and Language Modeling
Implements the NMT model of [Luong et al., 2015](https://arxiv.org/abs/1508.04025) from scratch in PyTorch. The RNNs are non-stacked.
- GRU-based **encoder** and **decoder**
- **Global attention** mechanism
- A full `seq2seqModel` with training loop, plus evaluation with pretrained weights

### Lab 2: Pretraining and Supervised Finetuning
Parameter-efficient fine-tuning of **Qwen3.5-0.8B-Base** on a free Colab T4.
- Load the model in 4-bit and attach **LoRA** adapters
- **Summarization**: fine-tune on French news articles ([`giuliadc/orangesum_5k`](https://huggingface.co/datasets/giuliadc/orangesum_5k)), then merge the adapter into the base weights
- **Tool calling**: teach the model to call functions with [`llamafactory/glaive_toolcall_en`](https://huggingface.co/datasets/llamafactory/glaive_toolcall_en), with assistant-only label masking

### Lab 3: Post-Training LLMs with DPO and GRPO
Preference optimization and reinforcement learning on **Qwen3-0.6B**, using `trl`.
- **Part I, DPO**: off-policy preference tuning on French data, with a LoRA policy model and a frozen reference model
- **Part II, GRPO**: on-policy RL on **GSM8K**. It uses format and correctness rewards, and compares the model before and after training on 200 test examples

## Getting started

The notebooks are designed for **Google Colab** (a free T4 GPU is enough). Click an *Open In Colab* badge above. Each notebook installs its own dependencies in its first cells.

To run locally, you need Jupyter and a CUDA GPU for labs 2 and 3:

```bash
git clone https://github.com/Jrejorio/altegrad.git
cd altegrad
pip install jupyter torch transformers peft trl bitsandbytes datasets nltk matplotlib tqdm
jupyter notebook
```

Lab 3 also uses a free [Weights & Biases](https://wandb.ai) account for experiment tracking.

## Stack

PyTorch · Hugging Face Transformers · PEFT (LoRA) · TRL (DPO, GRPO) · bitsandbytes · Weights & Biases · NLTK
