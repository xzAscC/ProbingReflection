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
```

## Getting Started

### Running Inference

To run inference with the original model:
```bash
python src/inference.py --model [MODEL_NAME] --output_dir [OUTPUT_PATH]
```

To run inference with the inserted model:
```bash
python src/inference.py --model [MODEL_NAME] --injection  --injection_layer [INJECTION_LAYER]  --injection_alpha [INJECTION_ALPHA] --output_dir [OUTPUT_PATH]
```

### Model Insertion

To insert vectors into a model:
```bash
python src/save_insert_model.py --model [MODEL_NAME] --output_dir [OUTPUT_DIR]
```


For more details, refer to the scripts in the `scripts` directory.