SHELL := /bin/bash
.PHONY: clean data lint requirements sync_data_to_s3 sync_data_from_s3

PYTHON_INTERPRETER = python3

ifeq (,$(shell which conda))
HAS_CONDA=False
else
HAS_CONDA=True
endif

requirements: 
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

model_rna: 
	$(PYTHON_INTERPRETER) odonto/models/rna/train.py

evaluate_models_rna: 
	$(PYTHON_INTERPRETER) odonto/models/rna/evaluate.py

predict_online_rna:
	$(PYTHON_INTERPRETER) odonto/models/online/rna/predict.py	

predict_batch_rna: 
	$(PYTHON_INTERPRETER) odonto/models/batch/rna/predict.py