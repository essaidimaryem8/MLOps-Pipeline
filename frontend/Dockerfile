# Use the prebuilt base image with dependencies
FROM essaidimaryem/ml-project:latest

# Set the working directory
WORKDIR /app

# Copy the application code
COPY app.py .

# Ensure the container runs as myuser
USER myuser

# Expose the port for Streamlit
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
