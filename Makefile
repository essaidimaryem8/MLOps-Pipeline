.PHONY: all quality test prepare_data train evaluate retrain docker_build docker_push setup

setup:
	pip install -r requirements.txt
	pip install -r backend/requirements.txt

all: quality test prepare_data train evaluate

quality:
	pycodestyle backend/main.py backend/model_pipeline/preprocessing.py backend/model_pipeline/training.py backend/model_pipeline/evaluation.py backend/model_pipeline/io.py tests/test_main.py --max-line-length=120 --ignore=E203,E266,E501,W503

test:
	pytest tests/test_main.py

prepare_data: setup
	python backend/main.py --prepare_data

train: setup
	python backend/main.py --train

evaluate: setup
	python backend/main.py --evaluate

retrain: setup
	python backend/main.py --retrain

docker_build:
	docker build -t essaidimaryem/ml-pipeline-backend:latest -f backend/Dockerfile .
	docker build -t essaidimaryem/ml-pipeline-frontend:latest -f frontend/Dockerfile .

docker_push:
	docker login -u $(DOCKER_USERNAME) -p $(DOCKER_PASSWORD)
	docker push essaidimaryem/ml-pipeline-backend:latest
	docker push essaidimaryem/ml-pipeline-frontend:latest
