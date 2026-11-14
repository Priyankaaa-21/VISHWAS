.PHONY: help download ita train evaluate compare gradcam rag-index app test slice3d-audit slice3d-split slice3d-pretrain milk10k-s3d

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

slice3d-audit:
	@if [ -d "src" ]; then $(PYTHON) src/ingest/audit_slice3d.py; else $(PYTHON) vishwas/src/ingest/audit_slice3d.py; fi

slice3d-split:
	@if [ -d "src" ]; then $(PYTHON) src/preprocess/make_slice3d_split.py; else $(PYTHON) vishwas/src/preprocess/make_slice3d_split.py; fi

slice3d-pretrain:
	@if [ -d "src" ]; then $(PYTHON) src/train/slice3d_pretrain.py; else $(PYTHON) vishwas/src/train/slice3d_pretrain.py; fi

milk10k-s3d:
	@if [ -d "src" ]; then $(PYTHON) src/train/train.py --model two_image_tabular_fusion --fold all --epochs 12 --img-size 128 --mitigation class_weighted --init-encoder checkpoints/slice3d_pretrain_effb0.pt --init-branches clinical --run-name e_cw_128_s3d; else $(PYTHON) vishwas/src/train/train.py --model two_image_tabular_fusion --fold all --epochs 12 --img-size 128 --mitigation class_weighted --init-encoder checkpoints/slice3d_pretrain_effb0.pt --init-branches clinical --run-name e_cw_128_s3d; fi

