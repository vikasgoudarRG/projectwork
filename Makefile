.PHONY: validate-all validate-env validate-raw validate-templates validate-sessions validate-join validate-labels validate-vocab validate-windows validate-baseline validate-report validate-docs clean-validation

validate-all:
	python -m scripts.validate_data run --task all

validate-env:
	python -m scripts.validate_data run --task env

validate-raw:
	python -m scripts.validate_data run --task raw

validate-templates:
	python -m scripts.validate_data run --task templates

validate-sessions:
	python -m scripts.validate_data run --task sessions

validate-join:
	python -m scripts.validate_data run --task join

validate-labels:
	python -m scripts.validate_data run --task labels

validate-vocab:
	python -m scripts.validate_data run --task vocab

validate-windows:
	python -m scripts.validate_data run --task windows

validate-baseline:
	python -m scripts.validate_data run --task baseline

validate-report:
	python -m scripts.validate_data run --task report

validate-docs:
	python -m scripts.validate_data run --task docs

clean-validation:
	rm -rf artifacts/validation/*

# DeepLog Training & Evaluation
.PHONY: train-key train-value detect evaluate visualize finetune full-pipeline clean-models

train-key:
	python -m scripts.run_training --task key

train-value:
	python -m scripts.run_training --task value

detect:
	python -m scripts.run_training --task detect

evaluate:
	python -m scripts.run_training --task eval

visualize:
	python -m scripts.run_training --task visual

finetune:
	python -m scripts.run_training --task online

full-pipeline:
	make train-key && make train-value && make detect && make evaluate && make visualize

clean-models:
	@python3 - <<'PY'
	import shutil, os
	for p in ['models','artifacts/training','artifacts/detection','artifacts/eval','artifacts/visual','artifacts/online','graphs']:
	    if os.path.exists(p):
	        shutil.rmtree(p)
	        print(f'Removed {p}')
	print('✓ Cleaned all model artifacts')
	PY
