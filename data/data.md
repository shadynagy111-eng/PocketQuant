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

