# ArXiv Paper Chronology Prediction — RoBERTa Classification Pipeline

This repository contains a solution for the ArXiv Paper Chronology task. The objective is to analyze the first sentences of two scientific papers and determine which one was published later based on textual cues.

## Table of Contents
- [Problem Description](#problem-description)
- [Solution Approach](#solution-approach)
- [Pipeline & Architecture](#pipeline--architecture)
- [Results](#results)
- [Dependencies](#dependencies)
- [Usage](#usage)

---

## Problem Description
The dataset is built from the arXiv open-access repository, containing the opening sentences of papers across 140 topics published between 2000 and 2025. The challenge is to build a model capable of recognizing implicit temporal markers—such as evolving trends, newly introduced buzzwords, or references to changing methodologies. In the test set, papers are paired with at least a 2-year publication gap, and the model must perform binary classification to identify the newer publication.

## Solution Approach
To capture the subtle semantic shifts and chronological context within scientific text, this solution relies on advanced Natural Language Processing (NLP). The task is framed as a text sequence classification problem, utilizing a pre-trained transformer model (RoBERTa) to extract deep contextual representations and learn the chronological relationship between the texts.

## Pipeline & Architecture
* **Data Preprocessing:** Handled via the Hugging Face `datasets` library, efficiently converting `pandas` DataFrames into optimal formats. Texts are tokenized and padded/truncated using `AutoTokenizer`.
* **Model:** A pre-trained transformer initialized via `AutoModelForSequenceClassification`. This allows the network to leverage its extensive prior language understanding and adapt it specifically to the task of temporal ordering.
* **Training:** The fine-tuning process is managed using the `Trainer` and `TrainingArguments` API from the `transformers` library, optimizing for binary classification accuracy.
* **Inference:** The model predicts the probability distribution for the test pairs and outputs the final binary labels indicating the later publication.

## Results
The fine-tuned transformer model successfully learned to identify temporal trends in scientific language, achieving the following score on the evaluation metric:

* **Final Accuracy: 0.7280**

## Dependencies
The following libraries are required to run the training and inference pipeline:
* Python 3.x
* `torch`
* `transformers` (Hugging Face)
* `datasets`
* `pandas`
* `scikit-learn`

```bash
pip install torch transformers datasets pandas scikit-learn
