IMAGE_NAME := recsys-linkedin

.PHONY: help build dashboard figuras jupyter shell clean

help: ## Exibe os comandos disponíveis
	@echo "=========================================================="
	@echo " Comandos do Projeto (Docker + Makefile no Manjaro/Arch)   "
	@echo "=========================================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

build: ## Constrói a imagem Docker isolada
	docker build -t $(IMAGE_NAME) .

dashboard: ## Inicia o Dashboard Streamlit (http://localhost:8501)
	docker run --rm -it -p 8501:8501 -v "$$(pwd)":/app -e PYTHONPATH=/app $(IMAGE_NAME) streamlit run app/dashboard.py --server.address=0.0.0.0 --server.port=8501

figuras: ## Executa o script de geração de figuras da EDA
	docker run --rm -it -v "$$(pwd)":/app $(IMAGE_NAME) python src/gerar_figuras_eda.py

jupyter: ## Inicia o Jupyter Lab (http://localhost:8888) para rodar os notebooks
	docker run --rm -it -p 8888:8888 -v "$$(pwd)":/app $(IMAGE_NAME) jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --IdentityProvider.token=''

shell: ## Abre um terminal bash interativo dentro do container
	docker run --rm -it -p 8501:8501 -v "$$(pwd)":/app $(IMAGE_NAME) /bin/bash

clean: ## Limpa arquivos temporários de cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
