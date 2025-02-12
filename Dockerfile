# syntax=docker/dockerfile:1

# Use the official Python image from the Docker Hub
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .


# Install the dependencies without caching to keep the image small
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

ENV PYTHONPATH="/app/src"

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
# CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["python", "src/app.py"]

