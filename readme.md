# California Housing Price Prediction API

This project is a FastAPI-based application that predicts housing prices in California based on various features. The application is containerized using Docker and includes monitoring with Prometheus and Grafana.

## Installation

### Prerequisites

- Docker
- Docker Compose

### Clone the Repository

```bash
git clone https://github.com/yourusername/california-housing.git
cd california-housing
```

### Build and Run the Docker Containers
1. Build the Docker images:
`docker-compose build`
2. Run the docker containers:
`docker-compose up -d`
3. The FastAPI application will be accessible at http://localhost:8000
4. Prometheus will be accessible at http://localhost:9090
5. Grafana will be accessible at http://localhost:3000


## Usage
1. Start the FastApi Server
```python src/app.py```

2. Open your browser and go to `http://127.0.0.1:8000/docs` to access the swagger API documentation.

3. Make a post request to `/predict` endpoint to get the prediction with the required fields

## Project Structure
- `src/`: Source code for the application
  - `app.py`: Main application file
  - `model.py`: Model loading and saving functions
  - `data_download.py`: Script to download and load data
- `models/`: Directory to store the trained models
- `data/`: Directory to store data files
- `requirements.txt`: List of dependencies
- `Dockerfile`: Dockerfile to build the Docker image
- `docker-compose.yml`: Docker Compose file to run the application and monitoring services
- `.gitignore`: Git ignore file to exclude unnecessary files

## Monitoring
  The application includes monitoring with Promethius and Grafana
  - Promethius is configfured to scrape metrics from the FastApi Application
  - Grafana is configured to visualize the metrics captured by Prometheus

## Deploying to AWS 
  To deploy to the Docker container to AWS, you can use AWS Elastic Beanstalk, ECS (Elastic Container Service) or EKS (Elastic Kubernetes Service). 

1. Install AWS Elastic Beanstalk CLI
```pip install awsebcli```

2. Initialize the EB application
```eb init -p docker your-app-name```

3. Create elastic beanstalk environment to deploy the application
```eb.create your-env-name```

4. Open the app in web browser
```eb open```

Replace yourusername, your-app-name, and your-env-name with appropriate names for your application and environment. This will deploy your Docker container to AWS Elastic Beanstalk and make your application accessible via a public URL.

License
This project is licensed under the MIT License.

