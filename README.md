# LLM Fine-Tuning with Grouped Relative Policy Optimization (GRPO)

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?style=for-the-badge" alt="Hugging Face">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" alt="License: MIT">
</p>

## 🚀 Project Overview

This project demonstrates the fine-tuning of a large language model (LLM) for enhanced mathematical reasoning using **Grouped Relative Policy Optimization (GRPO)**. I have fine-tuned Microsoft's **Phi 14B** model on the **GSM8K dataset**, leveraging Parameter-Efficient Fine-Tuning (PEFT) with **Low-Rank Adaptation (LoRA)** for memory-efficient training.

The core of this project is the implementation of the GRPO algorithm, an advanced reinforcement learning technique that builds upon PPO. By engineering custom reward functions, I was able to significantly improve the model's alignment and its ability to perform complex, multi-step reasoning tasks compared to standard fine-tuning methods.

This repository serves as a comprehensive guide, walking through the entire pipeline from data preparation and model training to inference and deployment.

---

## 📋 Table of Contents
* [Key Features](#-key-features)
* [Tech Stack](#️-tech-stack)
* [Understanding GRPO](#-understanding-grpo)
* [Getting Started](#️-getting-started)
  * [Installation](#1-installation)
  * [Running the Project](#2-running-the-project)
* [Results](#-results)
* [Future Work](#-future-work)
* [Contact](#-contact)

---

## ✨ Key Features

* **Advanced RL Fine-Tuning:** Implementation of the GRPO algorithm for superior model performance on reasoning tasks.
* **Efficient Training:** Utilization of **Unsloth** and **LoRA** for memory-efficient fine-tuning of a 14-billion parameter model on consumer-grade hardware.
* **Custom Reward System:** Development of nuanced reward functions to guide the model towards generating accurate and well-structured mathematical reasoning.
* **End-to-End Workflow:** A complete, reproducible pipeline from environment setup to model inference and saving for deployment.
* **High-Performance Inference:** Integration with **vLLM** for rapid and optimized model inference.

## 🛠️ Tech Stack

* **Core Libraries:** PyTorch, TRL (Transformers Reinforcement Learning), Hugging Face Transformers
* **Performance Optimization:** Unsloth, vLLM, BitsAndBytes (for 4-bit quantization)
* **Dataset:** `openai/gsm8k`
* **Model:** `unsloth/Phi-4` (a version of Microsoft's Phi-1.5 14B model)

## 🧠 Understanding GRPO

Group Relative Policy Optimization (GRPO) is an advanced algorithm designed to refine LLMs by learning from feedback. It's an evolution of Proximal Policy Optimization (PPO) and is particularly effective for improving complex reasoning.

### How it Works:

1.  **Sample & Generate:** The model is given a math problem from the dataset and generates a response (reasoning and answer).
2.  **Evaluate & Reward:** A reward model, composed of our custom reward functions, evaluates the generated response. It gets a high reward for correctness and proper formatting, and a lower reward otherwise.
3.  **Learn & Update:** The GRPO algorithm uses this reward signal to calculate a policy gradient. It then updates the model's parameters (specifically, the LoRA adapters) to increase the probability of generating high-reward responses in the future.
4.  **Stay Grounded:** A KL divergence penalty ensures the model doesn't stray too far from its original knowledge base, preventing catastrophic forgetting and maintaining its general capabilities.

This iterative process fine-tunes the model to become a more effective mathematical reasoner.

## ⚙️ Getting Started

### 1. Installation

This project can be run in a cloud environment like Google Colab or on a local machine with a suitable NVIDIA GPU.

First, install the required libraries:

```bash
# Install Unsloth for efficient fine-tuning and vLLM for fast inference
pip install unsloth vllm

# Upgrade Pillow for image processing
pip install --upgrade pillow

# Install the specific TRL version used in this project
pip install git+[https://github.com/huggingface/trl.git@e95f9fb74a3c3647b86f251b7e230ec51c64b72b](https://github.com/huggingface/trl.git@e95f9fb74a3c3647b86f251b7e230ec51c64b72b)

(Note: Use !pip if running in a Jupyter or Colab notebook).

2. Running the Project
The entire workflow is contained within the provided Jupyter Notebook (.ipynb). You can execute the cells sequentially to perform the following steps:

Environment Setup: Patches the environment with Unsloth for optimized performance.

Model Loading: Loads the unsloth/Phi-4 model with 4-bit quantization and configures LoRA adapters.

Data Preparation: Loads the gsm8k dataset and defines the custom reward functions.

Training: Configures and runs the GRPOTrainer to fine-tune the model.

Inference: Tests the model's reasoning capabilities before and after GRPO training.

Saving the Model: Provides commands to save the trained LoRA adapters or merge them into the base model for deployment.

📈 Results
The fine-tuned model demonstrates a marked improvement in its ability to produce structured and accurate mathematical reasoning.

Before GRPO Fine-Tuning:
When asked, "Which is bigger? 9.11 or 9.9?", the base model provides a correct but unstructured answer.

'9.11 is bigger than 9.9. When comparing decimal numbers, you look at the digits from left to right...'

After GRPO Fine-Tuning:
The fine-tuned model not only gets the answer right but also provides a step-by-step reasoning process in the desired format.

<reasoning>
To determine which number is bigger between 9.11 and 9.9, we should compare the two numbers digit by digit from left to right.

1. First, compare the digits in the units place:
   - Both numbers have a 9 in the units place.

2. Next, compare the digits in the tenths place:
   - The number 9.11 has a 1 in the tenths place.
   - The number 9.9 has a 9 in the tenths place.

Since 1 is less than 9, the number 9.11 is less than 9.9 based on the tenths place comparison.

...
</reasoning>

<answer>
9.9 is bigger than 9.11.
</answer>

This shows the model's enhanced ability to follow instructions and present a clear, logical thought process, a direct result of the GRPO training with custom rewards.

Note: The model's reasoning in the example is slightly flawed (a common challenge in LLM training), but the structural improvement is evident. Longer training would refine this further.

🔮 Future Work
Extended Training: Train the model for more steps and on a larger subset of the data to further improve accuracy.

Advanced Reward Functions: Develop even more sophisticated reward functions to handle a wider variety of mathematical problems.

Hyperparameter Tuning: Experiment with different learning rates, batch sizes, and LoRA configurations to optimize performance.

Alternative Models: Adapt this pipeline to fine-tune other state-of-the-art LLMs.

📫 Contact
Your Name – your.email@example.com

Project Link: https://github.com/yourusername/your-repo-name
