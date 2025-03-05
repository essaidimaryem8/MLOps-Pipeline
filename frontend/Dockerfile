FROM essaidimaryem/ml-project:latest

WORKDIR /app

COPY app.py .

# Ensure altair is installed (in case the base image misses it)
RUN pip install altair==4.2.2

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
