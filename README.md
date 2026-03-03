# PhenoXtract
PhenoXtract: combining Large Language Model and Knowledge Graph embedding to extract phenotypes from clinical descriptions

## Requirements

- **Python >= 3.9**
- **requirements.txt**
- `pip` and `venv` (or `conda`)

## Installation

### Option pip + venv
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```
### Option conda
```bash
conda create -n phenoxtract python=3.9 -y 
conda activate phenoxtract 
python -m pip install --upgrade pip
```

### Install dependecies
```bash
pip install -r ./scripts/requirements.txt
```

## `scripts/` contents

### Embedding preparation and ontology completion

1. Download from https://hpo.jax.org/data/ontology the ontology file: `hp.owl`
2. Optional: run `populate_missing_terms.py` (it requires UMLS API KEY)
3. Run `optimal_transport.py`
Used for **filling missing ontology synonyms and definitions** and for **embedding preparation**.

### PhenoXtract main pipeline
- `main.py`
- `concept_recognition.py`
- `entity_linking.py`  
These are the two main steps of the **PhenoXtract** approach:
1. concept recognition
2. entity linking

Launch `main.py` to test PhenoXtract
```bash
python main.py --api-key '{YOUR_OPENAI_API_KEY}' --text 'Proband has high temperature' --json-path './preprocess_json' --output-json-path ./output_json.json
```

### Benchmarking
- `benchmark.py`  
Runs the benchmark on the **OLIDA** and **mitochondrial** datasets.
- `benchmark_GSC.py`  
Runs the benchmark on the **GSC** dataset (folder `GSC+`).

### Configuration and constants
- `constants.py`  
Configuration/constants file. **Add your openAI API key here to run benchmark scripts**.
