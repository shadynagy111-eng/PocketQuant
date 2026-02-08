Here is a specific `README.md` file for your **notebooks** folder.

You can place this file inside the `notebooks/` directory to help anyone (including future you) understand the order and purpose of each script.

---

# 📓 PocketQuant Notebooks

This directory contains the Jupyter notebooks used to preprocess data, fine-tune lightweight models, and run inference for the **PocketQuant** project.

## 📂 File Inventory

| Notebook | Status | Description |
| --- | --- | --- |
| **`01_Data_Preprocessing.ipynb`** | 🟢 Ready | Loads the raw FinQA JSON data, linearizes tables into Markdown strings, and formats text for LLM training. |
| **`02_Train_Gemma_2B.ipynb`** | 🟡 Core | Fine-tunes **Gemma 2 (2B)** using Unsloth. Optimized for logic and reasoning tasks. |
| **`03_Train_Llama_3B.ipynb`** | 🟡 Core | Fine-tunes **Llama 3.2 (3B)** using Unsloth. Optimized for chat and instruction following. |
| **`04_Inference_&_Eval.ipynb`** | 🔵 Test | Loads the saved adapters (LoRA) to generate answers for unseen questions and calculate accuracy. |

---

## 🔄 Recommended Workflow

For the best results, run the notebooks in the following order:

### 1. Data Preparation (`01_Data_Preprocessing.ipynb`)

* **Input:** Raw `train.json`, `test.json` from Kaggle.
* **Process:**
* Joins `pre_text` and `post_text` lists.
* Converts JSON `table` rows into `| Column 1 | Column 2 |` format.
* Create the standardized Prompt Template (`### Context: ... ### Question: ...`).


* **Output:** A Hugging Face `Dataset` object ready for the trainer.

### 2. Fine-Tuning (`02_...` or `03_...`)

* **Input:** The processed dataset from Step 1.
* **Settings:**
* **Quantization:** 4-bit (to fit on 16GB T4 GPUs).
* **LoRA Rank:** 32 (High rank for better learning on small models).
* **Batch Size:** 8 (optimized for 2B/3B models).


* **Output:** `lora_model` directory containing the trained adapters.

### 3. Inference (`04_Inference_&_Eval.ipynb`)

* **Input:** The base model + your trained `lora_model` folder.
* **Process:**
* Merges the adapter with the base model on-the-fly.
* Runs generation on the **Test Set**.
* Compares the generated answer (e.g., "5.2%") with the ground truth.



---

## ⚠️ Important Usage Notes

### **1. Kaggle / Colab Setup**

All notebooks are designed to run on **Linux** environments with NVIDIA GPUs.

* **Accelerator:** Select **GPU T4 x2** or **P100**.
* **Internet:** Must be **ON** to download base models (Gemma/Llama) and the Unsloth library.

### **2. Library Installation**

Every notebook starts with the following install block. **Do not remove it**, as Kaggle environments reset after every session.

```python
%%capture
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install --no-deps "xformers<0.0.27" "trl<0.9.0" peft accelerate bitsandbytes

```

### **3. Hugging Face Login**

To use **Gemma** or **Llama**, you must have a Hugging Face token with read permissions.

* In the notebook, use:
```python
from huggingface_hub import login
login(token="YOUR_HF_TOKEN")

```




