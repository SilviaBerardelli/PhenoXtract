# PhenoXtract
PhenoXtract: combining Large Language Model and Knowledge Graph embedding to extract phenotypes from clinical descriptions

## Requirements

- **Python >= 3.9**
- **requirements.txt**

## `scripts/` contents

### Embedding preparation and ontology completion
- `optimal_transport.py`
- `populate_missing_terms.py`  
Used for **embedding preparation** and for **filling missing ontology synonyms and definitions**.

### PhenoXtract pipeline
- `concept_recognition.py`
- `entity_linking.py`  
These are the two main steps of the **PhenoXtract** approach:
1. concept recognition
2. entity linking

### Benchmarking
- `benchmark.py`  
Runs the benchmark on the **OLIDA** and **mitochondrial** datasets.
- `benchmark_GSC.py`  
Runs the benchmark on the **GSC** dataset (folder `GSC+`).

### Configuration and constants
- `constant.py`  
Configuration/constants file. **Add your openAI API key here**.
