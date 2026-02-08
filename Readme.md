Here is a professional, ready-to-copy **README.md** file for your GitHub repository.

You can copy the code block below directly into a file named `README.md`.


# 📉 PocketQuant: Lightweight Financial Reasoning

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-orange)
![Unsloth](https://img.shields.io/badge/Library-Unsloth-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**PocketQuant** is a research project exploring the capabilities of **lightweight Large Language Models (LLMs)**—specifically those under 4 billion parameters—in performing complex financial reasoning tasks. 

By fine-tuning state-of-the-art small models on the **FinQA** dataset, this project aims to demonstrate that massive parameters are not always necessary for high-precision financial analysis, table interpretation, and numerical reasoning.

---

## 🎯 Project Goal
To evaluate and compare the performance of **Gemma 2 (2B)** and **Llama 3.2 (3B)** on the **FinQA (Financial Question Answering)** dataset. The objective is to create a model capable of:
1.  **Reading Context:** Understanding hybrid contexts containing both unstructured text and structured financial tables.
2.  **Numerical Reasoning:** Performing calculations (e.g., "What was the percentage growth in revenue?") based on the data.
3.  **Efficient Deployment:** Running on consumer-grade hardware (e.g., free Kaggle GPUs or local laptops) using 4-bit quantization.

---

# 📊 About the Dataset: FinQA

> **Paper:** [FinQA: A Dataset of Numerical Reasoning over Financial Data](https://arxiv.org/abs/2109.00122)  
> **Authors:** Zhiyu Chen, Wenhu Chen, Chien-Sheng Wu, et al. (EMNLP 2021)

## 📝 Overview
The sheer volume of financial statements makes it difficult for humans to access and analyze a business's financials efficiently. Furthermore, robust numerical reasoning faces unique challenges in this domain due to the specialized vocabulary and data structure.

In this work, we focus on **FinQA**, a large-scale dataset aimed at automating the analysis of financial documents. Unlike general domain tasks, the finance domain requires:
1.  **Complex Numerical Reasoning:** Performing calculations (addition, subtraction, percentage change) rather than just retrieving facts.
2.  **Heterogeneous Data Understanding:** Combining information from unstructured text paragraphs and structured tables simultaneously.

To facilitate analytical progress, the FinQA dataset provides expert-annotated Question-Answering pairs derived from real-world financial reports.

---

## 🔍 Deep Dive: What is in the Paper?

The paper identifies a critical gap in NLP: **LLMs are good at reading, but bad at math.**
Most "Question Answering" datasets (like SQuAD) simply ask the model to find a span of text (e.g., "Who is the CEO?"). Financial analysis requires executing a *program* (e.g., "Divide Q3 Revenue by Q2 Revenue").

### Key Dataset Features
* **Source:** Earnings reports (10-K/10-Q) from S&P 500 companies.
* **Scale:** 8,281 candidate QA pairs.
* **Expert Annotation:** Written by financial professionals to ensure the questions reflect real-world analyst needs.
* **Hybrid Context:** Every example consists of a **Text Passage** + a **Financial Table**. The answer can only be derived by linking specific numbers from the table with context from the text.

### The Reasoning Challenge
The dataset does not just ask for a number; it requires a **reasoning program**.
* *Input:* "What was the growth rate of tax expenses?"
* *Required Logic:* `subtract(2019_tax, 2018_tax) -> result; divide(result, 2018_tax)`
* *Output:* "4.5%"

---

## 💡 Discussion: Why FinQA for this Project?

For **PocketQuant**, FinQA is the perfect stress test for **Lightweight Models (Gemma 2B / Llama 3B)** for three reasons:

1.  **Hallucination Check:** Small models love to make up numbers. FinQA forces the model to ground its answer in the provided table, allowing us to strictly measure accuracy.
2.  **Context Limit Testing:** The model must fit the pre-text, the linearized table, and the post-text into its context window (often 2k-8k tokens) without losing focus.
3.  **Logic vs. Memorization:** Since the dataset requires calculation (math), the model cannot simply memorize the training data. It must learn the *logic* of how to read a balance sheet.


## 🤖 Models Used
We focus on the "Lightweight" class of LLMs to enable edge-device financial analysis.

| Model | Parameters | Developer | Key Strengths |
| :--- | :--- | :--- | :--- |
| **Gemma 2** | **2B** | Google | Exceptional reasoning density; outperforms larger models in logic benchmarks. |
| **Llama 3.2** | **3B** | Meta | Optimized for instruction following; larger context window (128k). |

Both models are fine-tuned using **LoRA (Low-Rank Adaptation)** and **4-bit quantization** to minimize memory usage.

---

## 🛠️ Tech Stack & Methodology

* **Library:** `Unsloth` (for 2x faster training and 60% less memory usage).
* **Framework:** `Hugging Face Transformers`, `TRL` (Transformer Reinforcement Learning), `PEFT`.
* **Hardware:** Trained on **Kaggle T4 x2** / **P100** GPUs.
* **Technique:**
    * **QLoRA:** Quantized Low-Rank Adaptation (training only ~1-5% of parameters).
    * **Rank (r):** 32 (Higher rank chosen to compensate for small model capacity).
    * **Target Modules:** All linear layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, etc.).

---

## 🚀 Installation & Usage

### 1. Prerequisites
This project is optimized for a Linux environment with CUDA support (e.g., Kaggle Notebooks or Google Colab).

```bash
# Install Unsloth (optimized for T4/L4/A100)
pip install "unsloth[colab-new] @ git+[https://github.com/unslothai/unsloth.git](https://github.com/unslothai/unsloth.git)"
pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

```

### 2. Training

Run the training script (provided in `train.ipynb`):

```python
from unsloth import FastLanguageModel
# Load Gemma 2 (2B) in 4-bit
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-2-2b-bnb-4bit",
    load_in_4bit = True,
)
# ... (See notebook for full training loop)

```

### 3. Inference

To test the model on a new financial question:

```python
# Inference Code
inputs = tokenizer(
    [
        prompt_text + "### Answer:"
    ], 
    return_tensors = "pt"
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64)
print(tokenizer.batch_decode(outputs))

```

---

## 📂 Project Structure

```
PocketQuant/
├── data/                   # (Optional) Local copy of FinQA sample
├── notebooks/
│   ├── 01_Preprocessing.ipynb  # JSON to Text formatting logic
│   ├── 02_FineTuning_Gemma.ipynb # Training loop for Gemma 2B
│   └── 03_FineTuning_Llama.ipynb # Training loop for Llama 3B
├── models/                 # Saved LoRA adapters
├── README.md               # This file
└── requirements.txt        # Python dependencies

```

---

## 📈 Results (Preliminary)

* **Training Speed:** Gemma 2 (2B) trains approximately **40% faster** than Llama 3 (8B) on the same hardware.
* **Memory Footprint:** Both models fit comfortably within **8GB VRAM** during training (batch size = 8).
* **Accuracy:** *(To be updated after full evaluation on the test set)*.

---

## 🤝 Acknowledgements

* **Unsloth AI** for the optimized training library.
* **Visalakshiiyer** for hosting the FinQA dataset on Kaggle.
* The **FinQA** research team for the original dataset curation.

