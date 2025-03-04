# Use the prebuilt base image with dependencies
FROM essaidimaryem/ml-project:latest

# Set the working directory
WORKDIR /app

# Ensure /app is writable by root during build
RUN chmod -R 755 /app

# Copy the application code
COPY main.py .
COPY model_pipeline/ ./model_pipeline/

# Create /app/data directory and set ownership to myuser
RUN mkdir -p /app/data && chown -R myuser:myuser /app /app/data && chmod -R 755 /app /app/data

# Ensure the container runs as myuser
USER myuser

# Expose the port for FastAPI
EXPOSE 8000

# Run the FastAPI app with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
