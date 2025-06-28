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
