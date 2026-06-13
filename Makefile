lint:
	uv run ruff check --fix . && uv run ruff format .

upload-training:
	docker build -t us-central1-docker.pkg.dev/abiding-kingdom-491823-b0/predictive-maintenance-repo/predictive-maintenance:v1 .
	docker push us-central1-docker.pkg.dev/abiding-kingdom-491823-b0/predictive-maintenance-repo/predictive-maintenance:v1
	python kfp_pipeline.py --project abiding-kingdom-491823-b0 --pipeline-root gs://predictive-maintenance-test/kfp-pipeline --bucket-name predictive-maintenance-test