FROM python:3.9-slim

WORKDIR /app

RUN apt-get update -y && apt-get upgrade -y

# Create a non-root user
RUN useradd -m appuser

# Copy project files
COPY requirements.txt .

# Install dependencies using pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY models/model.pkl ./models/

# Change ownership of the application files
RUN chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]