FROM python:3.9-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Train the model when building the container
RUN python -m model.model

# Expose the port
EXPOSE 5000

# Command to run the API
CMD ["python", "-m", "api.app"]
