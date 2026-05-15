# Use an official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Create and set the working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browser and its OS dependencies
RUN playwright install --with-deps chromium

# Copy the rest of the application code
COPY . .

# Expose port (default for Flask/Gunicorn is 5000 or 8000 depending on config, Render assigns a dynamic PORT)
EXPOSE 5000

# Command to run on container start
CMD gunicorn app:app --bind 0.0.0.0:$PORT
