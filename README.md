# HELM Benchmark Optimization with DSPy

[![arXiv](https://img.shields.io/badge/arXiv-tbd-b31b1b.svg?style=for-the-badge)](https://arxiv.org/abs/tbd)
[![License](https://img.shields.io/github/license/stanfordmimi/helm-optimizer?style=for-the-badge)](LICENSE)

<img src="assets/fig.png" alt="Overview" width="700">

**Figure 1** | **HELM Optimizer Workflow**. Pipeline describing (a) DSPy-based prompt optimization for each model, and (b) performance analysis of baseline prompt vs DSPy prompt across models on the HELM leaderboard.

A comprehensive framework for optimizing language models on HELM benchmarks using DSPy (Declarative Self-improving Language Programs). This toolkit enables automated prompt optimization, leveraging benchmarks from the HELM (Holistic Evaluation of Language Models) ecosystem.

## Installation

```bash
# Clone the repository
git clone https://github.com/StanfordMIMI/helm-optimizer.git
cd helm-optimizer

# Install dependencies
pip install -r requirements.txt
```

## Basic Usage

**Configure settings** in `run.sh`:
```bash
scenarios=("medcalc_bench" "medec" "head_qa" "medbullets")
optimizers=("MIPROv2" "BootstrapFewShotWithRandomSearch")

model=openai/gpt-4o
api_base="your_api_base_here"
api_key="your_api_key_here"

max_bootstrapped_demos=3
max_labeled_demos=3
num_threads=1
```

**Run optimization**:
```bash
./run.sh
```

## 📊 Supported Benchmarks

| Benchmark | Description | Task Type | Metric |
|-----------|-------------|-----------|---------|
| **MedCalc-Bench** | Compute a specific medical value from a patient note | Computational reasoning | MedCalc Accuracy |
| **Medec** | Detect and correct errors in medical narratives | Classification | Medical Error Flag Accuracy |
| **HeadQA** | Medical knowledge testing | Question answering | Exact match |
| **Medbullets** | Medical knowledge testing	 | Question answering | Exact match |

## Optimization Parameters

- `--max_bootstrapped_demos`: Number of bootstrapped demonstrations
- `--max_labeled_demos`: Number of labeled demonstrations
- `--num_threads`: Parallel processing threads (default: 16)

## 📁 Project Structure

```
helm-optimizer/
├── main.py              # Main optimization script
├── scenarios.py         # Benchmark implementations
├── run.sh              # Batch optimization runner
├── requirements.txt    # Python dependencies
├── agents/            # Optimized DSPy agents
│   ├── medcalc_bench/
│   ├── head_qa/
│   ├── medbullets/
│   └── ...
└── README.md          # This file
```

## 📈 Results and Evaluation

Optimized agents are automatically saved to the `agents/` directory:

```
agents/
└── {scenario}/
    └── {model_name}/
        ├── MIPROv2.json
        └── BootstrapFewShotWithRandomSearch.json
```

## Creating Custom Scenarios

To add a new HELM benchmark:

1. **Implement the scenario class** in `scenarios.py`:
```python
class my_benchmark:
    def __init__(self, test_size=0.1, seed=42):
        self.test_size = test_size
        self.seed = seed
    
    @staticmethod
    def make_prompt(row):
        return f"Question: {row['question']}\nAnswer:"
    
    @staticmethod
    def metric(example, pred, trace=None):
        # Your evaluation metric
        return dspy.evaluate.metrics.answer_exact_match(example, pred, trace)
    
    def load_data(self):
        # Load and return trainset, valset
        pass
```

2. **Add to the scenarios list** in `run.sh`:
```bash
scenarios=("my_benchmark")
```

## 🙏 Acknowledgments

This repository is built using [DSPy](https://github.com/stanfordnlp/dspy) for language model optimization.

## 📎 Citation

If you find this repository useful for your work, please cite the following paper:

```bibtex
```
