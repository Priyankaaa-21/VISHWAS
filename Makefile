.PHONY: help download ita train evaluate compare gradcam rag-index app test

PYTHON := ./vishwas/.venv/bin/python3
PYTEST := ./vishwas/.venv/bin/pytest
STREAMLIT := ./vishwas/.venv/bin/streamlit

help:
	@echo "Vishwas Project Commands:"
	@echo "  make download    : Ingest & download images (defaults to --limit 50; pass LIMIT=all for full)"
	@echo "  make test        : Run test suite (including partition leakage tests)"
	@echo "  make ita         : Extract Individual Typology Angle (ITA) skin tone values"
	@echo "  make train       : Train classification model (supports baseline & fairness mitigations)"
	@echo "  make evaluate    : Evaluate fairness metrics (disparities across light/medium/dark)"
	@echo "  make compare     : Compare fairness and performance across different runs"
	@echo "  make gradcam     : Generate Grad-CAM explainability overlays"
	@echo "  make rag-index   : Index knowledge base documents for retrieval-grounded explanations"
	@echo "  make app         : Run Streamlit interactive diagnostic & fairness demo"

download:
	@if [ "$(LIMIT)" = "all" ]; then \
		$(PYTHON) vishwas/src/ingest/download_fitzpatrick.py; \
	elif [ -n "$(LIMIT)" ]; then \
		$(PYTHON) vishwas/src/ingest/download_fitzpatrick.py --limit $(LIMIT); \
	else \
		$(PYTHON) vishwas/src/ingest/download_fitzpatrick.py --limit 50; \
	fi

test:
	$(PYTEST) vishwas/tests/ -v

ita:
	$(PYTHON) vishwas/src/preprocess/skin_tone.py

train:
	$(PYTHON) vishwas/src/train/train.py

evaluate:
	$(PYTHON) vishwas/src/fairness/evaluate.py

compare:
	$(PYTHON) vishwas/src/fairness/compare_runs.py

gradcam:
	$(PYTHON) vishwas/src/explain/gradcam.py

rag-index:
	$(PYTHON) vishwas/src/rag/index_knowledge.py

app:
	$(STREAMLIT) run vishwas/src/app/app.py
