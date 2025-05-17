FROM python:3.10-slim

WORKDIR /app

# Copy requirements file first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the package files
COPY promptbuilder/ ./promptbuilder/
COPY README.md DEPLOYMENT.md LICENSE ./

# Install the package
RUN pip install -e .

# Create output directory
RUN mkdir -p /data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set entry point
ENTRYPOINT ["python", "-m", "promptbuilder.main"]
CMD ["generate", "--examples", "10", "--output", "/data/examples.jsonl"] 