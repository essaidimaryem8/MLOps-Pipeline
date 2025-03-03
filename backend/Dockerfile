# Use the prebuilt base image with dependencies
FROM essaidimaryem/ml-project:latest

# Set the working directory
WORKDIR /app

# Copy the application code
COPY main.py .
COPY model_pipeline/ ./model_pipeline/

# Ensure the container runs as myuser
USER myuser

# Expose the port for FastAPI
EXPOSE 8000

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
