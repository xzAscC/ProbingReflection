# ProbingReflection
Source Code of paper "From Emergence to Control: Probing and Modulating Self-Reflection in Language Models"

## Requirement

- transformers
- datasets
- vllm
- latex2sympy2_extended
- pylatexenc
- umap-learn

## Folder Architecture

```text
src: source code
    acc_length_rel.py: explore the relationship between length and acc
    math_grader.py: evaluation the math problem
    inference.py: inference with original model and inserted model
    save_insert_model.py: use vector to do insertion
    utils.py: other functions
asset: folder to save responses
models: folder to save model weights
scripts: bash file to inference