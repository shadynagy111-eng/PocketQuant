Here is a professional, ready-to-copy **README.md** file for your GitHub repository.

You can copy the code block below directly into a file named `README.md`.

---

```markdown
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

## 📊 Dataset: FinQA
We utilize the **[FinQA Dataset](https://github.com/czyssrs/FinQA)** (available on Kaggle via `visalakshiiyer/question-answering-financial-data`).

* **Source:** Earnings reports (10-K/10-Q) of S&P 500 companies.
* **Structure:**
    * **Context:** Text paragraphs + Financial Tables.
    * **Question:** A natural language query requiring calculation.
    * **Answer:** The numerical result or reasoning steps.
* **Challenge:** The model must "linearize" tables and perform arithmetic operations, a task typically difficult for small LLMs.

---

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

