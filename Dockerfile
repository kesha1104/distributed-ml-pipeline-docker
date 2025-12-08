FROM python:3.11-slim

# Working directory
WORKDIR /app

# Install Java 21 (required by PySpark)
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-21-jre-headless && \
    rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME for PySpark
ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-arm64
ENV PATH="$JAVA_HOME/bin:${PATH}"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY predict.py .
COPY ValidationDataset.csv .
COPY model_lr ./model_lr

# Default command when running the container
ENTRYPOINT ["python", "predict.py", "model_lr", "ValidationDataset.csv"]
