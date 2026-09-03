FROM python:3.11-slim

# Evita criação de arquivos .pyc e força stdout imediato no log
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instala dependências do Python
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Expõe as portas do Streamlit (8501) e Jupyter (8888)
EXPOSE 8501 8888

# O código e os dados serão montados via volume para flexibilidade
CMD ["streamlit", "run", "app/dashboard.py", "--server.address=0.0.0.0", "--server.port=8501"]
