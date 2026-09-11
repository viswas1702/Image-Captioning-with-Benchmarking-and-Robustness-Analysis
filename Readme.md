# Multimodal Image Captioning, Robustness Benchmarking & Model Attribution

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)

An end-to-end vision-language pipeline that implements a custom Vision Transformer (ViT) + GPT-2 image captioning model, benchmarks it against a zero-shot SmolVLM baseline, analyzes performance degradation under adversarial image occlusions, and trains a BERT-based sequence classifier for model output attribution.

---

## 📌 Key Highlights & Architecture

* **Custom ViT + GPT-2 Captioning Model:** Implemented an encoder-decoder architecture combining a Vision Transformer (ViT) visual feature extractor with an autoregressive GPT-2 language decoder.
* **Benchmarking vs. SmolVLM:** Evaluated generated captions against ground truth using standard NLP metrics (**BLEU-4, ROUGE-L, METEOR**), outperforming the zero-shot SmolVLM baseline.
* **Adversarial Robustness & Occlusion Analysis:** Stress-tested both architectures under progressive patch-level image occlusions (**10%, 50%, 80% masking**) to quantify semantic degradation.
* **BERT-based Model Discriminator:** Trained a sequence classifier on generated captions + perturbation metadata, achieving **97%+ accuracy** in identifying whether a caption originated from SmolVLM or the custom ViT-GPT2 model.

---

## 📊 Key Results & Benchmarks

| Model | BLEU-4 | ROUGE-L | METEOR | Notes |
| :--- | :---: | :---: | :---: | :--- |
| **SmolVLM (Zero-Shot Baseline)** | 0.0334 | - | - | Pretrained zero-shot baseline |
| **Custom ViT + GPT-2 (Ours)** | **0.0421** | Competitive | Competitive | Fine-tuned under 15GB VRAM constraints |
| **BERT Classifier Accuracy** | **97.0%+** | - | - | Model attribution & fingerprinting |

---

## 🛠️ Tech Stack & Requirements

* **Language:** Python 3.10+
* **Deep Learning Framework:** PyTorch (`torch`, `torchvision`)
* **Libraries:** Hugging Face (`transformers`, `datasets`, `accelerate`), `scikit-learn`, `nltk`, `rouge_score`, `pandas`, `numpy`
* **Hardware Environment:** Google Colab / CUDA-enabled GPU (T4 / V100 with ~15GB VRAM)

### Installation
```bash
pip install torch torchvision transformers datasets accelerate scikit-learn nltk rouge-score pandas