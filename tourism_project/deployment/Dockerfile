
# Minimal Python 3.9 base image
FROM python:3.9-slim

# Set container working directory
WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Create non-root user
RUN useradd -m user
USER user

# Expose Streamlit port
EXPOSE 8501

# Entrypoint
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
