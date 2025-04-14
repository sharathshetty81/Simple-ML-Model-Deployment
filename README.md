# Simple-ML-Model-Deployment
 Deploy a simple ML model using a pre-trained model and serve predictions through an API endpoint.

# ML Model Prediction API

This project demonstrates how to deploy a pre-trained machine learning model as a REST API using Flask and Docker.

## Features

- Pre-trained scikit-learn Random Forest model trained on the Iris dataset
- Flask REST API for making predictions
- Docker containerization for easy deployment
- Basic test suite
- Ready to deploy on any container hosting platform

## Quick Start

### Clone the Repository

```bash
git clone https://github.com/yourusername/ml-prediction-api.git
cd ml-prediction-api
```

### Using Docker Compose (Recommended)

```bash
docker-compose up --build
```

### Using Docker Directly

```bash
docker build -t ml-prediction-api .
docker run -p 5000:5000 ml-prediction-api
```

### Running Without Docker

```bash
pip install -r requirements.txt
python -m model.model  # Train and save the model
python -m api.app      # Start the API server
```

## API Endpoints

- `GET /health` - Check if the API is running
- `GET /info` - Get information about the model
- `POST /predict` - Make a prediction

### Example Prediction Request

```bash
curl -X POST http://localhost:5000/predict \
    -H "Content-Type: application/json" \
    -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

Example response:
```json
{
  "prediction": 0,
  "class_name": "setosa",
  "probabilities": {
    "setosa": 1.0,
    "versicolor": 0.0,
    "virginica": 0.0
  }
}
```
