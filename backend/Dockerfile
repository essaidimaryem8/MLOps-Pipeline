# Use the prebuilt base image with dependencies
FROM essaidimaryem/ml-project:latest

# Set the working directory
WORKDIR /app

# Copy the application code
COPY backend/main.py .
COPY backend/model_pipeline/ ./model_pipeline/

# Create /app/data directory and set ownership for /app/data only
RUN mkdir -p /app/data && \
    chown myuser:myuser /app/data && \
    chmod 755 /app/data

# Copy the data files into /app/data
COPY data/raw_data/churn-bigml-80.csv /app/data/
COPY data/raw_data/churn-bigml-20.csv /app/data/

# Ensure the container runs as myuser
USER myuser

# Expose the port for FastAPI
EXPOSE 8000

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
