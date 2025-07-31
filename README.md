# Medical Metaphors Corpus (MCC)

A comprehensive dataset of 792 annotated scientific conceptual metaphors spanning medical and biological domains.

## Overview

The Medical Metaphors Corpus (MCC) is the first openly released resource that captures metaphorical language across the breadth of medical and biological discourse. It addresses a critical gap between general-domain metaphor datasets and the specific needs of scientific NLP applications.

### Key Features

- **792 annotated sentences** with metaphorical expressions
- **82 distinct metaphor types** across medical and biological domains
- **24 unique target domains** and **38 unique source domains**
- **Binary metaphoricity labels** (metaphorical/literal)
- **Graded metaphoricity scores** (0-7 scale, where 0 = literal, 7 = highly metaphorical)
- **Source-target conceptual mappings** following Conceptual Metaphor Theory (CMT)

## Dataset Statistics

| Metric | Count | Percentage |
|--------|-------|------------|
| "Yes" (Metaphorical) | 353 | 44.57% |
| "No" (Literal) | 305 | 38.51% |
| Ties (Split decisions) | 134 | 16.90% |
| **Total sentences** | **792** | **100%** |


## Data Sources

The corpus aggregates metaphors from nine diverse sources across different discourse types:

| Source | Type | Count | Description |
|--------|------|-------|-------------|
| Van Rijn-van Tongeren (1997) | Literature | 455 | Medical metaphors from scientific articles |
| Camus (2009) | News | 19 | Cancer metaphors from The Guardian |
| Kaikarytė (2020) | News | 145 | Medical metaphors from UK news outlets |
| Semino et al. (2018) | Social Media | 27 | Patient forum discussions |
| Fereralda et al. (2022) | Social Media | 35 | Cancer patient stories |
| Cheded et al. (2022) | News | 35 | Medical metaphors in healthcare news |
| Gibbs Jr & Franks (2002) | Interviews | 50 | Cancer patient narratives |
| Sinnenberg et al. (2018) | Social Media | 40 | Diabetes discussions on Twitter |
| Metamia | Crowdsourced | 16 | User-contributed metaphors |

## File Structure

```
medical-metaphors-corpus/
├── README.md                          
├── m3c.csv                           # Main dataset file
├── code/
│   ├── calculate_llm_scores.py       # code for LLM evaluation with confidence weighting
│   └── example_query_llm.py          # Example LLM querying script
└── data/
    ├── llm_responses_claude.csv      # Claude evaluation results
    ├── llm_responses_deepseek.csv    # DeepSeek evaluation results
    ├── llm_responses_gpt4.csv        # GPT-4 evaluation results
    ├── llm_responses_o1preview.csv   # o1-preview evaluation results
    ├── llm_responses_o3mini.csv      # o3-mini evaluation results
    └── QualtricsSurveyResponses...csv # Raw human annotation data, from which to gather the silver standard
```


## Baseline LLM Performance

We evaluated five state-of-the-art LLMs on metaphor detection:

### Standard Metrics
| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|----| 
| o1-preview | 0.716 | 0.714 | 0.714 | 0.714 |
| Claude Opus 4 | 0.711 | 0.746 | 0.725 | 0.707 |
| o3-mini | 0.706 | 0.785 | 0.727 | 0.695 |
| DeepSeek | 0.683 | 0.745 | 0.702 | 0.673 |
| GPT-4 | 0.655 | 0.785 | 0.727 | 0.695 |

### Confidence-Weighted Metrics
| Model | wAccuracy | wPrecision | wRecall | wF1 |
|-------|-----------|------------|---------|-----|
| o1-preview | 0.758 | 0.716 | 0.714 | 0.716 |
| Claude Opus 4 | 0.755 | 0.756 | 0.721 | 0.705 |
| o3-mini | 0.752 | 0.799 | 0.706 | 0.690 |
| DeepSeek | 0.725 | 0.757 | 0.683 | 0.668 |
| GPT-4 | 0.690 | 0.776 | 0.655 | 0.626 |

## Usage

### Loading the Dataset

```python
import pandas as pd

# Load the main dataset
df = pd.read_csv('m3c.csv')

# Basic statistics
print(f"Total sentences: {len(df)}")
print(f"Metaphorical sentences: {df['binary_metaphoricity'].sum()}")
print(f"Mean metaphoricity score: {df['mean_metaphoricity_score'].mean():.2f}")
```

### Evaluation Scripts

#### 1. LLM Evaluation with Confidence Weighting

```bash
python code/calculate_llm_scores.py
```

This script:
- Reads Qualtrics survey responses
- Builds majority-vote lookup with confidence weights
- Evaluates LLM responses against human judgments
- Provides both standard and confidence-weighted metrics

#### 2. Querying LLMs for Metaphor Detection

```python
# See code/example_query_llm.py for a complete example
from openai import OpenAI

def llm_trial(text):
    prompt = (
        "In this task, you are asked to determine if the sentence is a metaphor or not. "
        "Respond only with **Yes** and **No**.\n\n"
        f"Question: \"{text}\"\n\n"
    )
    # ... (see full example in code/)
```

### Confidence-Weighted Evaluation

The corpus includes a novel confidence-weighted evaluation framework:

```python
# Calculate confidence weight
confidence = max(yes_votes, no_votes) / total_votes
weight = 2 * (confidence - 0.5) if confidence > 0.5 else 0.0

# Weighted accuracy
weighted_accuracy = sum(weights * correct_predictions) / sum(weights)
```



## Examples

### High-Rated Metaphors (Mean score > 5.0)
- *"It is inside the lungs that the virus turns nasty. It invades the millions of tiny air sacs in the lungs, causing them to become inflamed."*
- *"Three-step theory of invasion"* (about cell biology)

### Low-Rated Metaphors (Mean score < 1.0)
- *"Two of its main activities—of the plasma membrane—are selective transport of molecules into and out of the cell."*
- *"I have learned to let the little things go."* (cancer patient narrative)



## Ethical Considerations

- All data collected from publicly available sources
- No private medical information accessed
- Human annotation with informed consent
- No personally identifiable information included
- Compliant with fair use and fair dealing provisions for academic research



## License

This dataset is released under CC BY 4.0 for academic and research purposes.


