# MTSQIGA: A Novel Extractive Multi-document Text Summarization System using Quantum-Inspired Genetic Algorithm

Code repository for the manuscript [A Novel Extractive Multi-document Text Summarization System using Quantum-Inspired Genetic Algorithm](https).

Authors: Mohammad Mojrian and Seyed Abolghasem Mirroshandel (University of Guilan).

Contact: mojrian@msc.guilan.ac.ir

## Prerequisites

The code is developed with `python 3.6` and uses NLTK module for preprocessing step. For more requirements, please check `requirements.txt`.

- Python >= 3.6
- NLTK

## Installing NLTK data

We have used two corpora of NLTK in this code. To install them you should first install required packages:

```bash
make setup
```

Then you should install `stopwords` and `punkt` corpora of NLTK. To install them:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

## Run the main algorithm

To run the summarizer on a dataset, use command:

```bash
make serve
```

## Evaluation with ROUGE

The `evaluation` directory contains some code to evaluate the generated summary with [ROUGE](https://github.com/kavgan/ROUGE-2.0) toolkit and get a CSV output from the results. It includes codes to organize a directory based on ROUGE format as well as providing tools to utilize ROUGE on the generated summaries in terms of ROUGE-1, ROUGE-2, and ROUGE-SU4.
