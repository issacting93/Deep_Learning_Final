FROM python:3.9-slim

WORKDIR /app

# Install system dependencies needed for FAISS & PyTorch
RUN apt-get update && apt-get install -y libomp-dev gcc && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
# (.dockerignore handles excluding raw audio and venv)
COPY . .

# Expose port
EXPOSE 5050

# Run Gunicorn
CMD ["gunicorn", "-w", "4", "--timeout", "120", "-b", "0.0.0.0:5050", "app:app"]
